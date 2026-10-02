"""A master flat from panel frames: `seeingmon flat make`.

You light the lens evenly with a panel (see the runbook), record frames, and this module turns
them into the flat that `[survey] flat_file` loads: a 2-D `float32` image of the whole sensor,
scaled to a median of 1. The sky builder (`seeingmon.survey.flat_sky`) describes its result with
the same measures (`seeingmon.survey.flat_report`).

**Frames.** A source is a SER recording (the file that SharpCap or ASICap records) or a folder of
FITS files with one frame each. `--frames` can name several sources. Each one is a *set*: frames
taken with the light source in one position. A frame passes three tests, and the rest drop out.
Its mean level lies between 20% and 80% of full scale, so the sensor is neither near the bias nor
near saturation. The level lies within 5% of the median level of its set, so a flickering source
does not average in. The frame is not uneven, which means that the standard deviation of its
pixels is at most 30% of the mean (a wrong byte order, or a source that lights only part of the
lens, breaks this).

**Units.** A 14-bit camera in a 16-bit container holds the ADC value in the high bits, so every
count is a multiple of 4. The command finds out which scale a file uses. A file whose pixels share
low zero bits is a container. A file that declares an ADC depth (`ADCBITS`, or the depth of a SER
file) holds native counts of that depth. Any other file holds native counts of the depth that the
profile gives. `--full-scale` sets the full scale by hand.

**Bias.** The bias level comes off every frame. Bias frames are not necessary, and bias frames
with light in them (a lens that was not covered) do more harm than a good level does. The command
takes the level, in this order, from `--bias-level`, from bias frames that pass a check (a noise of
at most 3.5 counts per pixel, and a middle at most 0.5 counts brighter than the corners, in native
counts), and from the dark library (the bias of the sets with the readout mode and the gain of the
frames, interpolated to the sensor temperature). Bias frames that fail the check cost a warning,
and the next source in the order serves instead.

**Combination.** For each set, the frames are scaled to a common level (the median level above the
bias), the pixels are averaged with a sigma clip (a pixel more than `clip_sigma` standard
deviations from the mean of its frames drops out, and the mean is taken again), and the result is
divided by its median. The flat is the mean of the sets, divided by its median. Each frame is read
three times (the levels, the moments for the clip, and the clipped sum), and only one frame is in
memory at a time, apart from a few images of the size of a frame.

**Sets and the light source.** A light source has a gradient of its own, up to about 1% for a
phone screen, and the tilt of a flat from one set includes it. Turn the source by 180 degrees
between two sets, and the gradient of the source flips while the tilt of the optics and the sensor
stays. Half the sum of the two tilts is then the tilt of the optics and the sensor, half the
difference is the gradient of the source, and the mean of the two sets is a flat without the
gradient (to first order). The pixels cannot tell whether you turned the source, so
`--source-turned` says so, and the report calls the split by those names only on that word.
Without it, a plane of the quotient of the two sets above 0.3% in either direction is a warning:
the source drifted, or you turned it and did not say so.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock
from seeingmon.survey import flat_report

if TYPE_CHECKING:
    from seeingmon.survey.dark import DarkLibrary

FITS_SUFFIXES = frozenset({".fits", ".fit", ".fts"})
SATURATED_FRACTION = 0.98  # of full scale, from where a pixel counts as saturated
MIN_FLAT = 0.05  # a pixel below this (a dead one) takes this value, so that 1 / flat works
# The drop reasons, in the order that the report lists them.
LOW, HIGH, UNEVEN, FLICKER = "low", "high", "uneven", "flicker"

Frame2D = npt.NDArray[Any]
Float32Array = npt.NDArray[np.float32]
ByteOrder = Literal["little", "big"]


class FlatError(Exception):
    """The frames cannot make a flat. The message never names a path."""


# --- The options ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MakeOptions:
    """The thresholds of `make_flat`. The defaults follow the runbook.

    A frame passes when its mean level lies between `min_level_percent` and `max_level_percent` of
    full scale, within `flicker_percent` of the median level of its set, and the standard deviation
    of its pixels is at most `max_spread_percent` of its mean. `clip_sigma` is the sigma clip of the
    average. `min_frames` and `min_bias_frames` are the counts below which a warning goes out, and
    `max_saturated_fraction` is the share of saturated pixels above which one does. Bias frames fail
    their check when the noise of a pixel exceeds `bias_max_noise` or the middle exceeds the
    corners by `bias_max_gradient` (both in native counts). Two sets that you did not turn differ
    in tilt by more than `plane_limit_percent` across the width or the height: that is a warning.
    `full_scale` (in counts) sets the full scale by hand. `bin_factor` and `high_pass_px` (in
    binned pixels) set the report, `edge_margin_px` is the distance from an edge within which a dip
    counts as an edge artifact, and `center_xy` (sensor pixels) moves the optical center that the
    report measures from.
    """

    min_level_percent: float = 20.0
    max_level_percent: float = 80.0
    flicker_percent: float = 5.0
    max_spread_percent: float = 30.0
    clip_sigma: float = 3.0
    min_frames: int = 15
    min_bias_frames: int = 10
    max_saturated_fraction: float = 0.001
    bias_max_noise: float = 3.5
    bias_max_gradient: float = 0.5
    plane_limit_percent: float = 0.3
    edge_margin_px: float = flat_report.EDGE_MARGIN_PX
    full_scale: float | None = None
    bin_factor: int = 4
    high_pass_px: float = 40.0
    center_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.min_level_percent < self.max_level_percent <= 100:
            raise ValueError("the level limits must satisfy 0 <= min < max <= 100 percent")
        if self.flicker_percent <= 0 or self.max_spread_percent <= 0 or self.clip_sigma <= 0:
            raise ValueError("the flicker, spread, and clip limits must be positive")
        if self.bin_factor < 1 or self.high_pass_px <= 0:
            raise ValueError("the binning and the high-pass width must be positive")
        if self.full_scale is not None and self.full_scale <= 0:
            raise ValueError("the full scale must be positive")


@dataclass(frozen=True, slots=True)
class Geometry:
    """The sensor of the survey mode: its shape `(height, width)`, plate scale, and ADC depth."""

    shape: tuple[int, int]
    scale_arcsec_px: float
    adc_bits: int | None = None


# --- Sources of frames ----------------------------------------------------------------------


class FrameSource(Protocol):
    """A set of frames, read one at a time. `frame(i)` returns the 2-D pixels of frame `i`."""

    @property
    def count(self) -> int: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def declared_bits(self) -> int | None:
        """The ADC depth that the file declares for native counts, or `None`."""
        ...

    @property
    def temperature_c(self) -> float | None:
        """The mean sensor temperature that the files state, or `None`."""
        ...

    @property
    def gain(self) -> int | None:
        """The camera gain that the files state, or `None`."""
        ...

    def frame(self, index: int) -> Frame2D: ...

    def close(self) -> None: ...


class SerSource:
    """A SER recording of monochrome frames."""

    def __init__(self, path: Path, *, byte_order: ByteOrder | None = None) -> None:
        from seeingmon.recordings.ser import SerError, SerFile

        try:
            self._file = SerFile(path, byte_order=byte_order)
        except SerError as error:  # `SerFormatError` is a `SerError`
            raise FlatError(f"cannot read the recording: {error}") from None
        header = self._file.header
        if header.planes != 1:
            self._file.close()
            raise FlatError(
                "the recording has color planes, and the camera is monochrome: record in mono"
            )
        if self._file.frame_count == 0:
            self._file.close()
            raise FlatError("the recording holds no frame")
        self._declared = header.pixel_depth if 8 < header.pixel_depth < 16 else None

    @property
    def count(self) -> int:
        return int(self._file.frame_count)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._file.height), int(self._file.width))

    @property
    def declared_bits(self) -> int | None:
        return self._declared

    @property
    def temperature_c(self) -> float | None:
        return None

    @property
    def gain(self) -> int | None:
        return None

    def frame(self, index: int) -> Frame2D:
        return self._file.frame(index)

    def close(self) -> None:
        self._file.close()


class FitsSource:
    """A folder of FITS files, one frame in each.

    The files hold 16-bit integers (`BZERO` 32768), 8-bit integers, or floating-point numbers, and
    no compression, as the capture programs write them. A file that says `ROWORDER = 'BOTTOM-UP'`
    stores its rows from the bottom, and the source turns them so that the frame has the row order
    of the survey frames. The source reads the sensor temperature (`CCD-TEMP`), the gain (`GAIN`),
    and the ADC depth (`ADCBITS`) from the headers when they are there.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        from seeingmon.solvers import fitsio

        if not paths:
            raise FlatError("the folder holds no FITS file")
        self._paths = list(paths)
        self._flip: list[bool] = []
        temperatures: list[float] = []
        gains: list[int] = []
        bits: set[int] = set()
        shape: tuple[int, int] | None = None
        for path in self._paths:
            try:
                header = fitsio.read_header(path)
            except (OSError, fitsio.FitsError) as error:
                raise FlatError(f"cannot read a FITS file: {error}") from None
            if header.get("NAXIS") != 2:
                raise FlatError("a FITS file does not hold a 2-D image in its primary unit")
            this = (_as_int(header.get("NAXIS2")), _as_int(header.get("NAXIS1")))
            if shape is None:
                shape = this
            elif this != shape:
                raise FlatError("the FITS files differ in size")
            self._flip.append(str(header.get("ROWORDER", "")).upper().startswith("BOTTOM"))
            for key in ("CCD-TEMP", "SENSTEMP", "TEMPERAT"):
                value = header.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    temperatures.append(float(value))
                    break
            gain = header.get("GAIN")
            if isinstance(gain, int | float) and not isinstance(gain, bool):
                gains.append(round(gain))
            declared = header.get("ADCBITS")
            if isinstance(declared, int) and not isinstance(declared, bool):
                bits.add(declared)
        assert shape is not None
        self._shape = shape
        self._temperature = float(np.mean(temperatures)) if temperatures else None
        self._gain = Counter(gains).most_common(1)[0][0] if gains else None
        usable = {value for value in bits if 8 < value < 16}
        self._declared = usable.pop() if len(usable) == 1 else None

    @property
    def count(self) -> int:
        return len(self._paths)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def declared_bits(self) -> int | None:
        return self._declared

    @property
    def temperature_c(self) -> float | None:
        return self._temperature

    @property
    def gain(self) -> int | None:
        return self._gain

    def frame(self, index: int) -> Frame2D:
        from seeingmon.solvers import fitsio

        try:
            _, image = fitsio.read_image(self._paths[index])
        except (OSError, fitsio.FitsError) as error:
            raise FlatError(f"cannot read a FITS file: {error}") from None
        return image[::-1] if self._flip[index] else image

    def close(self) -> None:
        return None


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FlatError("a FITS header lacks the size of its image")
    return value


def open_frames(path: Path, *, byte_order: ByteOrder | None = None) -> FrameSource:
    """Open a SER file, a folder of FITS files, or one FITS file as a source of frames."""
    if path.is_dir():
        found = sorted(
            entry
            for entry in path.iterdir()
            if entry.is_file()
            and entry.suffix.lower() in FITS_SUFFIXES
            and not entry.name.startswith(".")
        )
        return FitsSource(found)
    if not path.is_file():
        raise FlatError("a frame source does not exist: give a SER file or a folder of FITS files")
    if path.suffix.lower() in FITS_SUFFIXES:
        return FitsSource([path])
    return SerSource(path, byte_order=byte_order)


# --- Units ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameUnits:
    """The scale of the counts in a file.

    `full_scale` is the count where the ADC saturates. `shift` is the number of bits that the
    file shifts native counts up (2 for 14-bit counts in a 16-bit container), so a native count of
    `n` appears as `n * 2 ** shift`.
    """

    full_scale: float
    shift: int

    @property
    def native_to_file(self) -> float:
        return float(2**self.shift)


def detect_units(
    frame: Frame2D,
    *,
    declared_bits: int | None,
    adc_bits: int | None,
    full_scale: float | None = None,
) -> FrameUnits:
    """Find the scale of the counts of a frame. See the module text for the rules."""
    if frame.dtype.kind == "f":
        if full_scale is None:
            raise FlatError("the pixels are floating-point numbers: pass the full scale in counts")
        return FrameUnits(float(full_scale), 0)
    if frame.dtype == np.uint8:
        return FrameUnits(float(full_scale or 255.0), 0)
    common = int(np.bitwise_or.reduce(frame, axis=None))
    zeros = (common & -common).bit_length() - 1 if common > 0 else 0
    if declared_bits is not None:
        return FrameUnits(float(full_scale or 2.0**declared_bits - 1.0), 0)
    if zeros >= 1:  # native counts in the high bits of 16
        return FrameUnits(float(full_scale or 65535.0), zeros)
    top = int(frame.max())
    bits = adc_bits if adc_bits is not None and top < 2**adc_bits else 16
    return FrameUnits(float(full_scale or 2.0**bits - 1.0), 0)


# --- Bias -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LibraryBias:
    """The bias level that the dark library gives for the frames, in native counts."""

    level_native: float
    description: str


@dataclass(frozen=True, slots=True)
class BiasInput:
    """The sources of the bias, in the order that `make_flat` prefers them.

    `level` is a scalar in the counts of the frames. `frames` are bias frames, which the command
    checks. `library` is the level that the dark library gives.
    """

    level: float | None = None
    frames: FrameSource | None = None
    library: LibraryBias | None = None


@dataclass(frozen=True, slots=True)
class BiasChoice:
    """What the command subtracts: a scalar level, or a master image with its mean level."""

    source: str
    level: float
    master: Float32Array | None
    note: str
    warnings: tuple[str, ...] = ()


def counts_text(level: float, units: FrameUnits) -> str:
    """A level in counts: native counts first, and the counts of the frames when they differ."""
    if units.shift == 0:
        return f"{level:,.1f} counts"
    native = level / units.native_to_file
    return f"{native:,.1f} native counts ({level:,.1f} in the counts of the frames)"


def _frame_as_float(raw: Frame2D) -> Float32Array:
    """A new `float32` array of the pixels, so that the caller may change it in place."""
    return np.array(raw, dtype=np.float32)


def _stream_moments(
    source: FrameSource,
    indices: Sequence[int],
    transform: Callable[[Frame2D, int], Float32Array],
) -> tuple[Float32Array, Float32Array]:
    """The per-pixel mean and standard deviation of transformed frames, one frame at a time.

    The sums run on the difference from the first frame, so the squares stay small.
    """
    reference: Float32Array | None = None
    total: Float32Array | None = None
    squares: Float32Array | None = None
    for index in indices:
        data = transform(source.frame(index), index)
        if reference is None or total is None or squares is None:
            reference = data
            total = np.zeros_like(data)
            squares = np.zeros_like(data)
            continue
        data -= reference
        total += data
        squares += data * data
    if reference is None or total is None or squares is None:
        raise FlatError("no frame to average")
    n = len(indices)
    mean = total / np.float32(n)
    variance = np.maximum(squares / np.float32(n) - mean * mean, np.float32(0.0))
    variance *= np.float32(n / max(n - 1, 1))
    mean += reference
    return mean, np.sqrt(variance)


def analyze_bias(source: FrameSource, units: FrameUnits) -> tuple[Float32Array, int, float, float]:
    """The master bias, its frame count, the noise per pixel, and the middle minus the corners.

    The noise and the difference are in native counts.
    """
    indices = list(range(source.count))
    mean, sigma = _stream_moments(source, indices, lambda raw, _: _frame_as_float(raw))
    noise = float(np.median(sigma)) / units.native_to_file if len(indices) > 1 else 0.0
    binned = flat_report.block_mean(mean, 4)
    center = ((binned.shape[1] - 1) / 2.0, (binned.shape[0] - 1) / 2.0)
    radius = flat_report.radius_map(binned.shape, center)
    corner = float(np.hypot(*center))
    middle = float(binned[radius <= 0.1 * corner].mean())
    corners = float(binned[radius >= 0.9 * corner].mean())
    return mean, len(indices), noise, (middle - corners) / units.native_to_file


def choose_bias(
    bias: BiasInput, units: FrameUnits, options: MakeOptions, shape: tuple[int, int]
) -> BiasChoice:
    """Pick what to subtract: checked bias frames, the scalar, or the level of the library."""
    warnings: list[str] = []
    rejected = False
    if bias.frames is not None:
        if bias.frames.shape != shape:
            raise FlatError(
                f"the bias frames are {bias.frames.shape[1]} x {bias.frames.shape[0]} pixels, "
                f"and the flat frames are {shape[1]} x {shape[0]}"
            )
        master, count, noise, gradient = analyze_bias(bias.frames, units)
        problems: list[str] = []
        if count < 2:
            problems.append("only one bias frame, which gives no noise")
        elif noise > options.bias_max_noise:
            problems.append(
                f"the noise is {noise:.1f} counts per pixel, over the limit of "
                f"{options.bias_max_noise:.1f} (a capped lens gives about 2)"
            )
        if gradient > options.bias_max_gradient:
            problems.append(
                f"the middle is {gradient:.1f} counts brighter than the corners, over "
                f"{options.bias_max_gradient:.1f}"
            )
        if count < options.min_bias_frames:
            warnings.append(
                f"Only {count} bias frames. Take {options.min_bias_frames} or more, or leave "
                "them out."
            )
        if not problems:
            level = float(master.mean(dtype=np.float64))
            return BiasChoice(
                "frames",
                level,
                master,
                f"{counts_text(level, units)}, from a master of {count} bias frames",
                tuple(warnings),
            )
        rejected = True
        warnings.append(
            "The bias frames hold light, so the command did not use them: "
            + "; ".join(problems)
            + ". Cover the lens for bias frames, or leave them out."
        )
    instead = " instead of the bias frames" if rejected else ""
    if bias.level is not None:
        return BiasChoice(
            "level",
            float(bias.level),
            None,
            f"{counts_text(float(bias.level), units)}, from --bias-level{instead}",
            tuple(warnings),
        )
    if bias.library is not None:
        level = bias.library.level_native * units.native_to_file
        return BiasChoice(
            "library",
            level,
            None,
            f"{counts_text(level, units)}, {bias.library.description}{instead}",
            tuple(warnings),
        )
    raise FlatError(
        "no bias level: record a dark set (seeingmon dark) for the dark library, or pass "
        "--bias-level N, or pass --bias with frames taken with the lens covered"
        + (f". The bias frames failed their check: {warnings[-1]}" if rejected else "")
    )


def library_bias(
    library: DarkLibrary,
    *,
    mode: str,
    gain: int,
    temperature_c: float | None,
    doubling_c: float,
) -> LibraryBias | None:
    """The bias level that the dark library gives for a readout mode and a gain.

    The level interpolates between the bias of the sets, in native counts, to the sensor
    temperature, and it stays at the first or the last set beyond them. Without a temperature, the
    level is the mean of the sets. The result is `None` when the library has no set of that mode
    and gain.
    """
    model = library.model(mode, gain, prior_doubling_c=doubling_c)
    if model is None:
        return None
    if temperature_c is None:
        level = float(np.mean([bias for _, bias in model.bias_points]))
        where = "the frames state no sensor temperature, so the mean of the sets"
    else:
        level = float(model.bias_dn(temperature_c))
        where = f"interpolated to {temperature_c:.1f} C"
    plural = "" if model.n_sets == 1 else "s"
    return LibraryBias(
        level,
        f"from the dark library ({model.n_sets} set{plural} of {mode} at gain {gain}, {where})",
    )


# --- The frames of a set --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameStat:
    """What the command measures of each frame before it combines them."""

    mean: float
    std: float
    saturated_fraction: float


def measure_frame(frame: Frame2D, units: FrameUnits) -> FrameStat:
    """The mean, the standard deviation, and the share of saturated pixels of a frame."""
    limit = SATURATED_FRACTION * units.full_scale
    return FrameStat(
        mean=float(frame.mean(dtype=np.float64)),
        std=float(frame.std(dtype=np.float64)),
        saturated_fraction=float(np.count_nonzero(frame >= limit)) / frame.size,
    )


@dataclass(frozen=True, slots=True)
class Selection:
    """Which frames of a set are used, and how many dropped for each reason."""

    used: tuple[int, ...]
    dropped: dict[str, int]


def select_frames(stats: Sequence[FrameStat], units: FrameUnits, options: MakeOptions) -> Selection:
    """Apply the level, spread, and flicker tests to the frames of one set."""
    dropped: Counter[str] = Counter()
    low = options.min_level_percent / 100.0 * units.full_scale
    high = options.max_level_percent / 100.0 * units.full_scale
    in_range: list[int] = []
    for index, stat in enumerate(stats):
        if stat.mean < low:
            dropped[LOW] += 1
        elif stat.mean > high:
            dropped[HIGH] += 1
        elif stat.std > options.max_spread_percent / 100.0 * stat.mean:
            dropped[UNEVEN] += 1
        else:
            in_range.append(index)
    if not in_range:
        return Selection((), dict(dropped))
    median = float(np.median([stats[index].mean for index in in_range]))
    used: list[int] = []
    for index in in_range:
        if abs(stats[index].mean / median - 1.0) > options.flicker_percent / 100.0:
            dropped[FLICKER] += 1
        else:
            used.append(index)
    return Selection(tuple(used), dict(dropped))


def describe_drops(dropped: dict[str, int], options: MakeOptions) -> str:
    """The reasons for the dropped frames as text, such as `2 below 20% of full scale`."""
    texts = {
        LOW: f"below {options.min_level_percent:g}% of full scale",
        HIGH: f"above {options.max_level_percent:g}% of full scale",
        UNEVEN: f"uneven (a spread above {options.max_spread_percent:g}% of the mean)",
        FLICKER: f"more than {options.flicker_percent:g}% from the median level (flicker)",
    }
    return ", ".join(
        f"{dropped[key]} {texts[key]}" for key in (LOW, HIGH, UNEVEN, FLICKER) if key in dropped
    )


@dataclass(frozen=True, slots=True)
class SetResult:
    """The report of one set of frames."""

    frames: int
    used: int
    dropped: dict[str, int]
    mean_level: float
    saturated_fraction: float
    noise_one_frame: float  # a fraction of the level above the bias, per pixel
    noise_mean: float  # the same for the mean of the used frames
    tilt: flat_report.Tilt


def _combine_set(
    source: FrameSource,
    selection: Selection,
    stats: Sequence[FrameStat],
    bias: BiasChoice,
    options: MakeOptions,
) -> tuple[Float32Array, float, float]:
    """The clipped mean of a set, its level above the bias, and the noise of a frame (counts)."""
    levels = {index: stats[index].mean - bias.level for index in selection.used}
    if min(levels.values()) <= 0:
        raise FlatError("a frame is at or below the bias level: check the bias level")
    target = float(np.median(list(levels.values())))
    offset: Float32Array | np.float32 = (
        bias.master if bias.master is not None else np.float32(bias.level)
    )

    def scaled(raw: Frame2D, index: int) -> Float32Array:
        data = _frame_as_float(raw)
        data -= offset
        data *= np.float32(target / levels[index])
        return data

    indices = list(selection.used)
    mean, sigma = _stream_moments(source, indices, scaled)
    noise = float(np.median(sigma))
    if len(indices) < 3:
        return mean, target, noise
    low = mean - np.float32(options.clip_sigma) * sigma
    high = mean + np.float32(options.clip_sigma) * sigma
    total = np.zeros(mean.shape, dtype=np.float32)
    kept = np.zeros(mean.shape, dtype=np.uint16)
    for index in indices:
        data = scaled(source.frame(index), index)
        keep = (data >= low) & (data <= high)
        total += np.where(keep, data, np.float32(0.0))
        kept += keep
    clipped = np.where(kept > 0, total / np.maximum(kept, 1), mean)
    return np.asarray(clipped, dtype=np.float32), target, noise


# --- The result -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MakeResult:
    """The flat and everything that the report says about it."""

    flat: Float32Array
    units: FrameUnits
    options: MakeOptions
    bias: BiasChoice
    sets: tuple[SetResult, ...]
    summary: flat_report.FlatSummary
    agreements: tuple[flat_report.SetAgreement, ...]
    split: flat_report.TiltSplit | None
    source_turned: bool
    noise_flat: float  # a fraction per pixel
    warnings: tuple[str, ...]
    elapsed_s: float | None = None


def make_flat(
    sets: Sequence[FrameSource],
    *,
    bias: BiasInput,
    geometry: Geometry,
    options: MakeOptions | None = None,
    source_turned: bool = False,
    clock: Clock | None = None,
) -> MakeResult:
    """Make a master flat from one or more sets of panel frames. See the module text.

    `source_turned` says that you turned the light source by 180 degrees between the sets. It
    changes what the report says and which warnings go out, and it never changes the flat.
    """
    cfg = options or MakeOptions()
    started = None if clock is None else clock.monotonic_ns()
    if not sets:
        raise FlatError("no frames: give --frames")
    for source in sets:
        if source.shape != geometry.shape:
            raise FlatError(
                f"the frames are {source.shape[1]} x {source.shape[0]} pixels, but the survey "
                f"mode has {geometry.shape[1]} x {geometry.shape[0]}: take the frames in the "
                "survey mode, with no region of interest"
            )
    declared = next((s.declared_bits for s in sets if s.declared_bits is not None), None)
    units = detect_units(
        sets[0].frame(0),
        declared_bits=declared,
        adc_bits=geometry.adc_bits,
        full_scale=cfg.full_scale,
    )
    choice = choose_bias(bias, units, cfg, geometry.shape)
    warnings: list[str] = list(choice.warnings)
    center = cfg.center_xy or ((geometry.shape[1] - 1) / 2.0, (geometry.shape[0] - 1) / 2.0)
    center_binned = flat_report.binned_position(center, cfg.bin_factor)

    reports: list[SetResult] = []
    total: Float32Array | None = None
    binned_sets: list[flat_report.FloatArray] = []
    noises: list[float] = []
    for number, source in enumerate(sets, start=1):
        label = f"set {number}" if len(sets) > 1 else "the frames"
        stats = [measure_frame(source.frame(i), units) for i in range(source.count)]
        selection = select_frames(stats, units, cfg)
        if not selection.used:
            raise FlatError(
                f"no frame of {label} passes the tests ({describe_drops(selection.dropped, cfg)}). "
                "Aim for 30 to 50% of full scale with a steady light, and check the byte order and "
                "whether the light covers the whole lens"
            )
        master, level, noise_counts = _combine_set(source, selection, stats, choice, cfg)
        used_stats = [stats[i] for i in selection.used]
        mean_level = float(np.mean([s.mean for s in used_stats]))
        saturated = float(np.mean([s.saturated_fraction for s in used_stats]))
        noise_one = noise_counts / level
        noise_mean = noise_one / math.sqrt(len(selection.used))
        master /= np.float32(np.median(master))
        binned = flat_report.block_mean(master, cfg.bin_factor)
        parts = flat_report.decompose(
            binned, center_xy=center_binned, high_pass_px=cfg.high_pass_px
        )
        reports.append(
            SetResult(
                frames=source.count,
                used=len(selection.used),
                dropped=selection.dropped,
                mean_level=mean_level,
                saturated_fraction=saturated,
                noise_one_frame=noise_one,
                noise_mean=noise_mean,
                tilt=flat_report.tilt_after_radial(binned, parts, center_xy=center_binned),
            )
        )
        percent = 100.0 * mean_level / units.full_scale
        if len(selection.used) < cfg.min_frames:
            warnings.append(
                f"Only {len(selection.used)} frames of {label} passed. Take {cfg.min_frames} or "
                "more, so that the noise of the flat stays low."
            )
        if saturated > cfg.max_saturated_fraction:
            warnings.append(
                f"{100 * saturated:.2f}% of the pixels of {label} are saturated, over "
                f"{100 * cfg.max_saturated_fraction:.1f}%. Dim the light or shorten the exposure."
            )
        if not cfg.min_level_percent <= percent <= cfg.max_level_percent:
            warnings.append(f"The mean level of {label} is {percent:.0f}% of full scale.")
        binned_sets.append(binned)
        noises.append(noise_mean)
        if total is None:
            total = master
        else:
            total += master
        del master

    assert total is not None
    flat = total / np.float32(len(noises))
    del total
    flat /= np.float32(np.median(flat))
    dead = int(np.count_nonzero(flat < MIN_FLAT))
    if dead:
        np.maximum(flat, np.float32(MIN_FLAT), out=flat)
        warnings.append(f"{dead} pixels read at or below zero, and the flat holds a floor there.")

    noise_flat = math.sqrt(sum(n * n for n in noises)) / len(noises)
    summary = flat_report.summarize_flat(
        flat_report.block_mean(flat, cfg.bin_factor),
        factor=cfg.bin_factor,
        sensor_shape=geometry.shape,
        scale_arcsec_px=geometry.scale_arcsec_px,
        center_xy=center_binned,
        high_pass_px=cfg.high_pass_px,
        edge_margin_px=cfg.edge_margin_px,
        # a quiet flat keeps more of the depth of a narrow shadow under a lighter smoothing
        smooth_sigma=0.5 if noise_flat / cfg.bin_factor <= 0.0015 else 1.0,
    )
    agreements = tuple(
        flat_report.compare_sets(
            binned_sets[0],
            binned_sets[number],
            factor=cfg.bin_factor,
            center_xy=center_binned,
            high_pass_px=cfg.high_pass_px,
            noise_first=noises[0],
            noise_second=noises[number],
        )
        for number in range(1, len(binned_sets))
    )
    split = flat_report.split_tilts(reports[0].tilt, reports[1].tilt) if len(reports) > 1 else None
    if len(sets) == 1:
        warnings.append(
            "The tilt may include the gradient of your light source, up to about 1% for a phone "
            "screen. Take a second set with the source turned by 180 degrees, and pass both sets "
            "with --frames and --source-turned."
        )
    elif not source_turned:
        for number, agreement in enumerate(agreements, start=2):
            plane = agreement.plane
            if max(abs(plane.width_percent), abs(plane.height_percent)) > cfg.plane_limit_percent:
                warnings.append(
                    f"Set 1 and set {number} differ in tilt ({flat_report.tilt_text(plane)}), by "
                    f"more than {cfg.plane_limit_percent:g}%, and you did not say that you turned "
                    "the source. If the source drifted between the sets, the tilt of the flat is "
                    "unreliable. If you turned it by 180 degrees, pass --source-turned."
                )
    elapsed = None
    if clock is not None and started is not None:
        elapsed = (clock.monotonic_ns() - started) / NS_PER_S
    return MakeResult(
        flat=flat,
        units=units,
        options=cfg,
        bias=choice,
        sets=tuple(reports),
        summary=summary,
        agreements=agreements,
        split=split,
        source_turned=source_turned,
        noise_flat=noise_flat,
        warnings=tuple(warnings),
        elapsed_s=elapsed,
    )


# --- The report -----------------------------------------------------------------------------


def format_make_report(result: MakeResult, *, name: str | None = None) -> list[str]:
    """The plain-text report of `make_flat`. It names no path of the machine."""
    height, width = result.flat.shape
    factor = result.options.bin_factor
    lines = [f"Flat from panel frames: {width} x {height} pixels, median 1."]
    many = len(result.sets) > 1
    for number, report in enumerate(result.sets, start=1):
        label = f"Set {number}" if many else "Frames"
        percent = 100.0 * report.mean_level / result.units.full_scale
        lines.append(
            f"{label}: {report.used} of {report.frames} frames used. Mean level "
            f"{report.mean_level:,.0f} counts, {percent:.1f}% of the full scale of "
            f"{result.units.full_scale:,.0f} counts."
        )
        if report.dropped:
            lines.append(f"  Dropped: {describe_drops(report.dropped, result.options)}.")
        lines.append(
            f"  Noise per pixel: {100 * report.noise_one_frame:.2f}% in one frame, "
            f"{100 * report.noise_mean:.2f}% in the mean. "
            f"Saturated pixels: {100 * report.saturated_fraction:.3f}%."
        )
    lines.append(f"Bias: {result.bias.note}.")
    lines.append(
        f"Noise of the flat: {100 * result.noise_flat:.2f}% per pixel, "
        f"{100 * result.noise_flat / factor:.2f}% at {factor} x {factor} binning."
    )
    if many and result.split is not None:
        lines.append("Tilt of each set after the radial part:")
        for number, report in enumerate(result.sets, start=1):
            lines.append(f"  set {number}: {flat_report.tilt_text(report.tilt)}")
        stays, turns = result.split.stays, result.split.turns
        if result.source_turned:
            lines.append(
                "Tilt of the optics and the sensor (half the sum of sets 1 and 2, which stays "
                f"when the source turns): {flat_report.tilt_text(stays)}"
            )
            lines.append(
                "Gradient of the light source in set 1 (half the difference, which turns with "
                f"the source): {flat_report.tilt_text(turns)}"
            )
        else:
            lines.append(f"Half the sum of sets 1 and 2: {flat_report.tilt_text(stays)}")
            lines.append(f"Half the difference of sets 1 and 2: {flat_report.tilt_text(turns)}")
            lines.append(
                "  Half the sum is the tilt of the optics and the sensor, and half the difference "
                "is the gradient of the light source, only if you turned the source by 180 "
                "degrees between the sets. Pass --source-turned to say that you did."
            )
        for number, agreement in enumerate(result.agreements, start=2):
            lines.append(f"Quotient of set 1 over set {number}:")
            lines.extend(flat_report.agreement_lines(agreement))
        lines.append(
            "  The plane of the quotient shows the gradient of the light source only if you "
            "turned or moved the source between the sets. If you did not, it shows how far the "
            "source drifted between them."
        )
    lines.extend(flat_report.profile_lines(result.summary.profile))
    lines.append(
        flat_report.tilt_line(result.summary.tilt, label="Tilt of the flat after the radial part")
    )
    if many:
        lines.append(
            "  The flat is the mean of the sets. A gradient of the source that turned between "
            "the sets drops out of the mean to first order, and one that did not stays in the tilt."
        )
    else:
        lines.append(
            "  The tilt includes the gradient of your light source. A second set with the source "
            "turned by 180 degrees (a second --frames) separates the two."
        )
    lines.extend(flat_report.shadow_lines(result.summary.shadows))
    lines.extend(
        flat_report.edge_artifact_lines(
            result.summary.edge_artifacts, margin_px=result.options.edge_margin_px
        )
    )
    if result.elapsed_s is not None:
        lines.append(f"Time: {result.elapsed_s:.1f} s.")
    lines.extend(f"Warning: {warning}" for warning in result.warnings)
    if name is not None:
        lines.append(f"Wrote {name}. Set flat_file in the [survey] table to its path.")
    return lines
