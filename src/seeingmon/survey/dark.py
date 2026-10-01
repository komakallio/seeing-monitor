"""The dark model, and the library of dark sets that it comes from.

The camera has no cooler and no lens cap, so the dark signal of a survey frame depends on the
sensor temperature, and you cover the camera yourself to measure it. `seeingmon dark` records a
set of frames (see `seeingmon.survey.dark_session`) and adds it to a `DarkLibrary`. The
pipeline reads the library to subtract the dark level from the sky, and to mask the hot pixels.

**A dark set.** A set holds the master dark (the per-pixel median of the dark frames, in ADC
counts) and the numbers that the pipeline needs: the bias level (the median of bias frames that
the same session took at the shortest exposure), the exposure, the mean sensor temperature, the
time, and the read noise. A set lives in one FITS file under `<calibration>/darks/`. The master
dark is the image of the primary unit. A binary table in the first extension lists the hot
pixels: the pixels that the master dark shows more than `hot_sigma` robust sigmas above their
neighbors.

**The model.** The dark rate (counts per second per pixel, bias removed) follows the temperature
as `rate = rate_ref * 2 ** ((T - 20 C) / doubling)`. The research notes give about 0.2 e-/s per
native pixel at 20 C and a doubling about every 6 C. `fit_dark_model` fits the rate at 20 C and
the doubling temperature to the sets of one readout mode and gain by least squares on the
logarithm of the rate. With one set, or sets within a few degrees of each other, the doubling
temperature stays at its prior. The bias level interpolates linearly between the sets.

**The sky needs only the level.** The sky brightness uses the median of the frame, so it needs
the median dark level (`DarkModel.level_dn`), not a per-pixel map. The hot-pixel list keeps the
detector from mistaking a hot pixel for a star.

**`dark_due`.** The library is due for a new set when it holds no set within a tolerance of the
current sensor temperature that is also newer than six months. `health` reports the condition.

**Time.** Nothing here reads a clock. The functions that need the time take `now_ns`.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, utc_ns_to_datetime, utc_ns_to_iso
from seeingmon.solvers import fitsio
from seeingmon.store.layout import DataLayout, write_atomic
from seeingmon.survey import _scipy
from seeingmon.survey.rawdata import clipped_mean, robust_level

log = logging.getLogger("seeingmon.survey")

CALIBRATION_DIRNAME = "calibration"
DARKS_DIRNAME = "darks"
LIBRARY_FORMAT = 1
KIND = "dark"
REFERENCE_TEMPERATURE_C = 20.0
DEFAULT_DOUBLING_C = 6.0  # the research notes: the dark current doubles about every 6 C
DEFAULT_TOLERANCE_C = 3.0
DEFAULT_MAX_AGE_DAYS = 183.0  # six months
SECONDS_PER_DAY = 86_400.0
HOT_TABLE = "HOTPIX"
_NAME = re.compile(r"^dark-\d{8}T\d{6}Z-[a-z0-9_]+-g\d+(?:-\d+)?\.fits$")
_RATE_FLOOR_DN_PER_S = 1e-6

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


class DarkError(Exception):
    """The dark library or a dark session failed."""


# --- The sets ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DarkSet:
    """The summary of one dark set in the library.

    All levels are in native ADC counts. `dark_dn` is the median of the master dark, so it
    includes the bias. `temperature_c` is the mean sensor temperature of the dark frames, and
    `t_utc_ns` is the time of the middle of the set.
    """

    name: str
    mode: str
    gain: int
    exposure_s: float
    temperature_c: float
    temperature_spread_c: float
    t_utc_ns: int
    n_frames: int
    n_bias_frames: int
    bias_dn: float
    dark_dn: float
    read_noise_dn: float
    adc_bits: int
    width_px: int
    height_px: int
    n_hot_pixels: int

    @property
    def rate_dn_per_s(self) -> float:
        """The median dark rate above the bias, in counts per second per pixel."""
        return (self.dark_dn - self.bias_dn) / self.exposure_s

    def age_s(self, now_ns: int) -> float:
        """The age of the set in seconds."""
        return (now_ns - self.t_utc_ns) / NS_PER_S

    def to_header(self) -> fitsio.Header:
        return {
            "SMKIND": KIND,
            "SMFORMAT": LIBRARY_FORMAT,
            "SMMODE": self.mode,
            "GAIN": self.gain,
            "EXPTIME": self.exposure_s,
            "SENSTEMP": self.temperature_c,
            "TEMPSPRD": self.temperature_spread_c,
            "SMUTCNS": self.t_utc_ns,
            "DATE-OBS": utc_ns_to_iso(self.t_utc_ns, digits=0),
            "NFRAMES": self.n_frames,
            "NBIAS": self.n_bias_frames,
            "BIASDN": self.bias_dn,
            "DARKDN": self.dark_dn,
            "RDNOISE": self.read_noise_dn,
            "ADCBITS": self.adc_bits,
            "NHOT": self.n_hot_pixels,
        }

    @classmethod
    def from_header(cls, name: str, header: fitsio.Header) -> DarkSet:
        """Read a set from the header of its file. Raises `DarkError` for another kind of file."""
        if header.get("SMKIND") != KIND:
            raise DarkError(f"{name} is not a dark set")
        try:
            return cls(
                name=name,
                mode=str(header["SMMODE"]),
                gain=_as_int(header["GAIN"]),
                exposure_s=_as_float(header["EXPTIME"]),
                temperature_c=_as_float(header["SENSTEMP"]),
                temperature_spread_c=_as_float(header["TEMPSPRD"]),
                t_utc_ns=_as_int(header["SMUTCNS"]),
                n_frames=_as_int(header["NFRAMES"]),
                n_bias_frames=_as_int(header["NBIAS"]),
                bias_dn=_as_float(header["BIASDN"]),
                dark_dn=_as_float(header["DARKDN"]),
                read_noise_dn=_as_float(header["RDNOISE"]),
                adc_bits=_as_int(header["ADCBITS"]),
                width_px=_as_int(header["NAXIS1"]),
                height_px=_as_int(header["NAXIS2"]),
                n_hot_pixels=_as_int(header["NHOT"]),
            )
        except KeyError as error:
            raise DarkError(f"{name} lacks the header card {error.args[0]}") from None


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DarkError("a header card that should hold a whole number does not")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DarkError("a header card that should hold a number does not")
    return float(value)


def master_median(
    frames: Sequence[npt.NDArray[np.uint16]], *, strip_rows: int = 128
) -> npt.NDArray[np.uint16]:
    """The per-pixel median of frames of one size, rounded to whole counts.

    The function works in strips of rows, so the memory that it needs beyond the frames stays
    small. An even count averages the two middle values.
    """
    if not frames:
        raise ValueError("no frames")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("the frames differ in size")
    out = np.empty(shape, dtype=np.uint16)
    for start in range(0, shape[0], strip_rows):
        stack = np.stack([frame[start : start + strip_rows] for frame in frames]).astype(np.float32)
        out[start : start + strip_rows] = np.rint(np.median(stack, axis=0)).astype(np.uint16)
    return out


def find_hot_pixels(
    master: npt.NDArray[np.uint16] | npt.NDArray[np.float32],
    *,
    sigma: float = 6.0,
    min_excess_dn: float = 3.0,
    neighborhood_px: int = 9,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """The hot pixels of a master dark: where it rises above its neighborhood.

    The function subtracts the local mean over `neighborhood_px` pixels, so a slow glow does not
    count. A pixel is hot when its excess exceeds `sigma` robust sigmas of the excess of all
    pixels, and also `min_excess_dn` counts. It returns the columns, the rows, and the excess.
    """
    image = np.asarray(master, dtype=np.float32)
    excess = image - _scipy.uniform_filter(image, neighborhood_px)
    _, noise = robust_level(excess)
    threshold = max(sigma * noise, min_excess_dn)
    rows, columns = np.nonzero(excess > threshold)
    return (
        columns.astype(np.int32),
        rows.astype(np.int32),
        excess[rows, columns].astype(np.float32),
    )


# --- The library -------------------------------------------------------------------------


def library_version(sets: Sequence[DarkSet]) -> str:
    """A short identifier of a collection of sets: the same sets give the same text."""
    text = ";".join(sorted(f"{item.name}:{item.t_utc_ns}" for item in sets))
    return f"darks-{zlib.crc32(text.encode()):08x}" if sets else "darks-none"


class DarkLibrary:
    """The folder of dark sets. It reads the headers when you ask, so it holds no stale state.

    `DarkLibrary.from_layout(layout)` puts the folder at `<data>/calibration/darks`. Opening a
    library creates nothing: the folder appears when the first set arrives.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)

    @classmethod
    def from_layout(cls, layout: DataLayout) -> DarkLibrary:
        """The library in the calibration folder of the data directory."""
        return cls(layout.root / CALIBRATION_DIRNAME / DARKS_DIRNAME)

    @property
    def directory(self) -> Path:
        return self._directory

    def sets(self) -> tuple[DarkSet, ...]:
        """Every set in the folder, oldest first. A file that is not a valid set is skipped."""
        found: list[DarkSet] = []
        if not self._directory.is_dir():
            return ()
        for path in sorted(self._directory.glob("dark-*.fits")):
            if not _NAME.match(path.name):
                continue
            try:
                found.append(DarkSet.from_header(path.name, fitsio.read_header(path)))
            except (DarkError, fitsio.FitsError, OSError) as error:
                log.warning("the dark library skips %s: %s", path.name, error)
        found.sort(key=lambda item: (item.t_utc_ns, item.name))
        return tuple(found)

    def sets_for(self, mode: str, gain: int | None = None) -> tuple[DarkSet, ...]:
        """The sets of one readout mode, and of one gain when you give it."""
        return tuple(
            item
            for item in self.sets()
            if item.mode == mode and (gain is None or item.gain == gain)
        )

    def version(self) -> str:
        """A short identifier of the sets in the folder (see `library_version`)."""
        return library_version(self.sets())

    def model(
        self,
        mode: str,
        gain: int,
        *,
        prior_doubling_c: float = DEFAULT_DOUBLING_C,
        now_ns: int | None = None,
        max_age_s: float | None = None,
    ) -> DarkModel | None:
        """The dark model of a readout mode and gain, or `None` when the library has no set.

        With `now_ns` and `max_age_s`, sets older than that stay out of the fit, unless all of
        them are older.
        """
        sets = self.sets_for(mode, gain)
        if now_ns is not None and max_age_s is not None:
            recent = [item for item in sets if item.age_s(now_ns) <= max_age_s]
            sets = tuple(recent) if recent else sets
        if not sets:
            return None
        return fit_dark_model(sets, prior_doubling_c=prior_doubling_c)

    def nearest(self, mode: str, gain: int, temperature_c: float) -> DarkSet | None:
        """The set of a mode and gain whose temperature is closest to `temperature_c`."""
        sets = self.sets_for(mode, gain)
        if not sets:
            return None
        return min(sets, key=lambda item: (abs(item.temperature_c - temperature_c), -item.t_utc_ns))

    def load_master(self, dark_set: DarkSet) -> npt.NDArray[np.uint16]:
        """The master dark image of a set, in native counts."""
        _, image = fitsio.read_image(self._directory / dark_set.name)
        return np.asarray(image, dtype=np.uint16)

    def hot_pixels(
        self, dark_set: DarkSet
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """The columns, rows, and excess (in counts) of the hot pixels of a set."""
        table = fitsio.read_table(self._directory / dark_set.name)
        return (
            np.asarray(table["X"], dtype=np.int32),
            np.asarray(table["Y"], dtype=np.int32),
            np.asarray(table["EXCESS"], dtype=np.float32),
        )

    def hot_pixel_mask(self, mode: str, gain: int, temperature_c: float) -> BoolArray | None:
        """The hot-pixel mask of the set nearest to a temperature, or `None` with no set."""
        chosen = self.nearest(mode, gain, temperature_c)
        if chosen is None:
            return None
        mask = np.zeros((chosen.height_px, chosen.width_px), dtype=np.bool_)
        columns, rows, _ = self.hot_pixels(chosen)
        mask[rows, columns] = True
        return mask

    def add_set(
        self,
        master: npt.NDArray[np.uint16],
        *,
        mode: str,
        gain: int,
        exposure_s: float,
        temperature_c: float,
        temperature_spread_c: float,
        t_utc_ns: int,
        n_frames: int,
        n_bias_frames: int,
        bias_dn: float,
        read_noise_dn: float,
        adc_bits: int,
        dark_dn: float | None = None,
        hot_sigma: float = 6.0,
        hot_min_excess_dn: float = 3.0,
    ) -> DarkSet:
        """Add a set: find its hot pixels, and write the file atomically. Returns the set.

        `dark_dn` is the mean level of the dark frames in counts. Leave it out to take the
        clipped mean of the master dark, which rounds each pixel to a whole count.
        """
        if exposure_s <= 0 or n_frames < 1:
            raise ValueError("a dark set needs a positive exposure and at least one frame")
        columns, rows, excess = find_hot_pixels(
            master, sigma=hot_sigma, min_excess_dn=hot_min_excess_dn
        )
        height, width = master.shape
        level = clipped_mean(master) if dark_dn is None else dark_dn
        name = self._free_name(mode, gain, t_utc_ns)
        summary = DarkSet(
            name=name,
            mode=mode,
            gain=gain,
            exposure_s=exposure_s,
            temperature_c=temperature_c,
            temperature_spread_c=temperature_spread_c,
            t_utc_ns=t_utc_ns,
            n_frames=n_frames,
            n_bias_frames=n_bias_frames,
            bias_dn=bias_dn,
            dark_dn=level,
            read_noise_dn=read_noise_dn,
            adc_bits=adc_bits,
            width_px=width,
            height_px=height,
            n_hot_pixels=int(columns.size),
        )
        payload = fitsio.image_bytes(master, header=summary.to_header()) + fitsio.table_bytes(
            {"X": columns, "Y": rows, "EXCESS": excess}, extname=HOT_TABLE
        )
        path = write_atomic(self._directory / name, payload)
        # Return the set as the file reads back: a header card keeps 15 significant digits.
        return DarkSet.from_header(name, fitsio.read_header(path))

    def _free_name(self, mode: str, gain: int, t_utc_ns: int) -> str:
        stamp = utc_ns_to_datetime(t_utc_ns).strftime("%Y%m%dT%H%M%SZ")
        safe_mode = re.sub(r"[^a-z0-9_]", "_", mode.lower())
        base = f"dark-{stamp}-{safe_mode}-g{gain}"
        name = f"{base}.fits"
        number = 1
        while (self._directory / name).exists():
            number += 1
            name = f"{base}-{number}.fits"
        return name


# --- The model ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DarkModel:
    """The dark level of one readout mode and gain as a function of the sensor temperature.

    `rate_ref_dn_per_s` is the median dark rate at `reference_c`. `bias_points` are the bias
    levels of the sets, sorted by temperature. `doubling_fitted` says whether the sets fixed the
    doubling temperature or the prior stands. `rms_log2` is the scatter of the fit in doublings.
    """

    mode: str
    gain: int
    reference_c: float
    rate_ref_dn_per_s: float
    doubling_c: float
    doubling_fitted: bool
    rms_log2: float | None
    bias_points: tuple[tuple[float, float], ...]
    n_sets: int
    version: str

    def rate_dn_per_s(self, temperature_c: float) -> float:
        """The dark rate above the bias, in counts per second per pixel."""
        return self.rate_ref_dn_per_s * math.pow(
            2.0, (temperature_c - self.reference_c) / self.doubling_c
        )

    def bias_dn(self, temperature_c: float) -> float:
        """The bias level: linear between the sets, and constant beyond the first and the last."""
        temperatures = [point[0] for point in self.bias_points]
        levels = [point[1] for point in self.bias_points]
        return float(np.interp(temperature_c, temperatures, levels))

    def level_dn(self, temperature_c: float, exposure_s: float) -> float:
        """The median level of a dark frame: the bias plus the dark rate times the exposure."""
        return self.bias_dn(temperature_c) + self.rate_dn_per_s(temperature_c) * exposure_s


def fit_dark_model(
    sets: Sequence[DarkSet],
    *,
    prior_doubling_c: float = DEFAULT_DOUBLING_C,
    min_span_c: float = 4.0,
    reference_c: float = REFERENCE_TEMPERATURE_C,
) -> DarkModel:
    """Fit the dark model to sets of one readout mode and gain.

    With sets that span at least `min_span_c` and at least two positive rates, the fit solves
    `log2(rate) = a + (T - reference_c) / doubling` by least squares. A doubling temperature
    outside 2 C to 20 C counts as a failed fit, and so does a rate that does not rise with the
    temperature. Otherwise the doubling temperature stays at `prior_doubling_c`, and the rate at
    the reference temperature comes from the sets with that prior.
    """
    if not sets:
        raise ValueError("no sets")
    modes = {(item.mode, item.gain) for item in sets}
    if len(modes) != 1:
        raise ValueError("the sets must share one readout mode and gain")
    mode, gain = next(iter(modes))
    temperatures = np.array([item.temperature_c for item in sets], dtype=np.float64)
    rates = np.array([item.rate_dn_per_s for item in sets], dtype=np.float64)
    positive = rates > _RATE_FLOOR_DN_PER_S
    doubling = prior_doubling_c
    fitted = False
    rms: float | None = None
    rate_ref: float
    if positive.sum() >= 2 and float(np.ptp(temperatures[positive])) >= min_span_c:
        x = temperatures[positive] - reference_c
        y = np.log2(rates[positive])
        slope, intercept = np.polyfit(x, y, 1)
        if slope > 0 and 2.0 <= 1.0 / slope <= 20.0:
            doubling = float(1.0 / slope)
            rate_ref = float(2.0**intercept)
            residual = y - (intercept + slope * x)
            rms = float(np.sqrt(np.mean(residual**2)))
            fitted = True
    if not fitted:
        scaled = np.maximum(rates, _RATE_FLOOR_DN_PER_S) * 2.0 ** (
            -(temperatures - reference_c) / doubling
        )
        rate_ref = float(np.median(scaled))
    points: dict[float, list[float]] = {}
    for item in sets:
        points.setdefault(round(item.temperature_c, 3), []).append(item.bias_dn)
    bias_points = tuple((t, float(np.mean(levels))) for t, levels in sorted(points.items()))
    return DarkModel(
        mode=mode,
        gain=gain,
        reference_c=reference_c,
        rate_ref_dn_per_s=rate_ref,
        doubling_c=doubling,
        doubling_fitted=fitted,
        rms_log2=rms,
        bias_points=bias_points,
        n_sets=len(sets),
        version=library_version(sets),
    )


# --- dark_due ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DarkStatus:
    """Whether the library needs a new set, and why.

    `nearest` is the set closest to the current temperature among the sets that count, and
    `gap_c` is its distance in degrees. `newest_age_days` is the age of the newest set.
    """

    due: bool
    reason: str
    nearest: DarkSet | None
    gap_c: float | None
    newest_age_days: float | None


def dark_status(
    library: DarkLibrary,
    temperature_c: float | None,
    now_ns: int,
    *,
    mode: str | None = None,
    gain: int | None = None,
    tolerance_c: float = DEFAULT_TOLERANCE_C,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
) -> DarkStatus:
    """Judge the library at a sensor temperature. See `dark_due` for the rule."""
    sets = [
        item
        for item in library.sets()
        if (mode is None or item.mode == mode) and (gain is None or item.gain == gain)
    ]
    if not sets:
        return DarkStatus(True, "the library holds no dark set", None, None, None)
    newest = max(sets, key=lambda item: item.t_utc_ns)
    newest_age_days = newest.age_s(now_ns) / SECONDS_PER_DAY
    recent = [item for item in sets if item.age_s(now_ns) / SECONDS_PER_DAY <= max_age_days]
    if not recent:
        return DarkStatus(
            True,
            f"the newest set is {newest_age_days:.0f} days old, over the limit of "
            f"{max_age_days:.0f} days",
            None,
            None,
            newest_age_days,
        )
    if temperature_c is None:
        return DarkStatus(False, "the camera reports no temperature", None, None, newest_age_days)
    nearest = min(recent, key=lambda item: abs(item.temperature_c - temperature_c))
    gap = abs(nearest.temperature_c - temperature_c)
    if gap > tolerance_c:
        return DarkStatus(
            True,
            f"no recent set within {tolerance_c:.1f} C of {temperature_c:.1f} C "
            f"(the nearest is {gap:.1f} C away)",
            nearest,
            gap,
            newest_age_days,
        )
    return DarkStatus(False, "a recent set covers the temperature", nearest, gap, newest_age_days)


def dark_due(
    library: DarkLibrary,
    temperature_c: float | None,
    now_ns: int,
    *,
    mode: str | None = None,
    gain: int | None = None,
    tolerance_c: float = DEFAULT_TOLERANCE_C,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
) -> bool:
    """Whether the library needs a new dark set at the current sensor temperature.

    The answer is `True` when the library holds no set that is within `tolerance_c` of
    `temperature_c` and no older than `max_age_days` (six months by default). Without a
    temperature, only the age counts. Pass `mode` and `gain` to look at one readout setting.
    """
    return dark_status(
        library,
        temperature_c,
        now_ns,
        mode=mode,
        gain=gain,
        tolerance_c=tolerance_c,
        max_age_days=max_age_days,
    ).due


# --- Is the camera covered? --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DarkCheckOptions:
    """The limits that decide whether a frame is dark.

    The median may rise above the bias by `rate_factor` times the expected dark rate, and by at
    least `min_rate_e_per_s`. The robust sigma may exceed the expected noise (read noise and the
    shot noise of the level) by `noise_factor`. At most `max_tail_fraction` of the pixels may
    lie more than `tail_sigma` sigmas above the median.
    """

    rate_factor: float = 3.0
    min_rate_e_per_s: float = 0.5
    noise_factor: float = 1.5
    tail_sigma: float = 8.0
    max_tail_fraction: float = 0.001


@dataclass(frozen=True, slots=True)
class DarkCheck:
    """The result of `check_dark_frame`. `reason` is empty when the frame is dark."""

    ok: bool
    level_dn: float
    sigma_dn: float
    excess_rate_e_per_s: float
    tail_fraction: float
    reason: str


def check_dark_frame(
    data: npt.NDArray[np.float32] | npt.NDArray[np.uint16],
    *,
    bias_dn: float,
    exposure_s: float,
    e_per_adu: float,
    read_noise_e: float,
    expected_rate_e_per_s: float,
    options: DarkCheckOptions | None = None,
) -> DarkCheck:
    """Decide whether a frame is dark: nothing lights the sensor, and the camera is covered.

    `data` is in native counts. `bias_dn` is the median of a bias frame from the same session.
    A frame passes when its median sits near the bias level (the dark current explains the rise),
    its spread matches the noise that the level implies, and it holds no stars: very few pixels
    stand far above the median.
    """
    cfg = options or DarkCheckOptions()
    median, sigma = robust_level(data)
    level = clipped_mean(data)
    excess_e = (level - bias_dn) * e_per_adu
    rate = excess_e / exposure_s
    limit = max(cfg.rate_factor * expected_rate_e_per_s, cfg.min_rate_e_per_s)
    expected_sigma_e = math.sqrt(read_noise_e**2 + max(excess_e, 0.0))
    sigma_e = sigma * e_per_adu
    flat = np.asarray(data).reshape(-1)
    step = max(1, flat.size // 1_000_000)
    sample = flat[::step].astype(np.float32)
    tail = float(np.mean(sample > np.float32(median + cfg.tail_sigma * max(sigma, 1e-3))))
    reasons: list[str] = []
    if rate > limit:
        reasons.append(
            f"the level is {rate:.2f} e-/s above the bias, over the limit of {limit:.2f}"
        )
    if sigma_e > cfg.noise_factor * expected_sigma_e:
        reasons.append(
            f"the noise is {sigma_e:.1f} e-, over {cfg.noise_factor * expected_sigma_e:.1f} e-"
        )
    if tail > cfg.max_tail_fraction:
        reasons.append(f"{100 * tail:.2f}% of the pixels stand far above the median")
    return DarkCheck(
        ok=not reasons,
        level_dn=level,
        sigma_dn=sigma,
        excess_rate_e_per_s=rate,
        tail_fraction=tail,
        reason="; ".join(reasons),
    )
