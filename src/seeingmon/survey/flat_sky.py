"""A flat from the night sky: `seeingmon flat build`.

**The idea.** The camera is fixed to the ground and points at the pole, so the sky turns about the
middle of the frame at 15 arcsec per second. The mean of many frames in the *sensor* frame holds
three things. It holds the flat times the mean sky. It holds the gradient of the sky that is fixed
to the ground (the airglow against altitude, the glow of light pollution), which is fixed to the
sensor too. And it holds the azimuthal mean about the pole of the structure that is fixed on the
sky (faint stars, nebulosity), because the rotation averages the rest away. A mount that tracks the
sky could not do this. The result has the vignetting to a fraction of a percent and the shadows of
the dust, and it has no tilt: the sky cannot tell a tilt of the flat from a gradient of the sky
itself, so the flat holds none. A tilt of the optics of about 1% stays in the frames. A panel flat
(`seeingmon flat make`) gives the tilt, and `--base-flat` combines the two (see
`seeingmon.survey.flat_base`).

**The frames.** The survey frames that `core` keeps as FITS files (every tenth long frame and the
frames of events, see `seeingmon.survey.framefile`) are the input. A frame is *usable* when the
Sun is below -18 degrees, the Moon is down or less than 25% lit, the cloud fraction is under 0.1,
the transparency is at least 0.95, and the sky level lies within 10% of the median level of the
chosen frames. Event frames stay out (the `KEPT` card names them). Every threshold is an option.
The cloud fraction and the transparency come from the header (`CLOUDFRC` and `TRANSP`), and the
Sun and the Moon come from the time of the frame and the `[site]` of the configuration, because
the header carries no position. A frame that lacks an input is *unchecked* and stays out, unless
`accept_unchecked` lets it in.

**One frame.** The command subtracts the dark the way the survey pipeline does: the level of the
dark model for the sensor temperature and the exposure, and the per-pixel master dark of the set
that matches the temperature, when the library has one (the report says which). It detects the
stars (`seeingmon.survey.detect`) and masks them with a radius that grows with their flux, the
saturated pixels, the hot pixels, the frame edge, and a large disk around Polaris, whose halo makes
a bright ring at the radius of its orbit (the brightest star is Polaris). It divides the frame by
its own sky level, a sigma-clipped median of the pixels that stay. It adds the masked frame into a
running sum, a sum of squares, and a count at 4 x 4 binning (the pixels that stay in each block).
Only one frame is in memory at a time.

**The accumulator.** The system keeps few frames (about 24 in a clear night, and the files expire
after 7 days), so a run can add its sums to a file (`--accumulator`). The file holds the sums, the
counts, and the time and the roll of every frame already added, so that a later run adds only the
new frames. A frame that expires from the folder stays in the sum. The file is written to a
temporary name and renamed, so a crash cannot corrupt it.

**The flat.** The mean `M` is the sum over the count, scaled to a median of 1. The radial part is
the azimuthal mean of `M` about the optical center, with the plane of `M` divided out first (a
steep gradient of the sky would leave its slope in the rings at the corners otherwise). The smooth
rest of `M` over the radial part, a Gaussian of 40 binned pixels, holds the tilt and the sky
gradients, and the flat leaves it out. The fine part is `M` over the radial part and the rest. The
flat is the radial part times the fine part, scaled to a median of 1 and spread over the sensor
with bilinear interpolation. This is the recipe of the sky average with a better edge: the radial
part comes from `M` itself, so the Gaussian cannot bias it where the vignetting is steepest.

**A base flat.** With `--base-flat FILE` the command divides the mean sky by a flat that you took
with a panel, and reports what changed: the vignetting, the new shadows, and the plane. It writes
nothing without `--update`. With `--update` it writes the base flat times a correction that holds
only the changes over their limits, and never the plane, so the tilt comes from the base.

**Checks.** The roll of a frame is the Earth rotation angle at its time. The report gives the
coverage: 360 degrees minus the largest gap between the rolls. A flat from fewer than 20 frames or
less than 60 degrees of roll gets a warning, because the rotation has not averaged the structure of
the sky. The ring check compares the mean of `M` at the radius of the orbit of Polaris with a smooth
baseline. A bump of more than 0.3% means that the mask around Polaris is too small.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, Clock, utc_ns_to_iso
from seeingmon.survey import flat_base, flat_report
from seeingmon.survey.framefile import (
    FrameMeta,
    parse_frame_header,
    read_frame_fits,
    read_frame_header,
)

if TYPE_CHECKING:
    from seeingmon.profile import Profile
    from seeingmon.scheduler.config import SiteConfig
    from seeingmon.survey.config import SurveyConfig
    from seeingmon.survey.dark import DarkLibrary, DarkModel, DarkSet
    from seeingmon.survey.detect import Detections, DetectOptions

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

FITS_SUFFIXES = frozenset({".fits", ".fit", ".fts"})
ACCUMULATOR_FORMAT = 1
EDGE_PX = 8  # pixels at the frame edge stay out, as in the sky level of the pipeline
SATURATION_FRACTION = 0.98
# A bright star's wing reaches this many pixels for each cube root of its flux over the sky level
# of a pixel. The wing of a lens falls about as the cube of the distance, so a star masks the
# distance at which its wing reaches a thousandth of the sky.
HALO_PX_PER_CUBE_ROOT = 1.0
MAX_HALO_PX = 150.0
POLARIS_FLUX_RATIO = 3.0  # Polaris is the brightest star by at least this factor
MIN_POLARIS_FIXES = 8
MIN_POLARIS_SPREAD_DEG = 90.0
PREFLIGHT_STRIDE = 8  # the cheap sky level reads every 8th pixel in each direction
DARK_EXPOSURE_TOLERANCE = 0.05

# The reasons that a frame stays out, in the order that the tests run, with their text.
UNREADABLE = "unreadable"
NO_TIME = "no_time"
TIME_INVALID = "time_invalid"
WRONG_SIZE = "wrong_size"
SHORT = "short"
EVENT = "event"
NO_CLOUD = "no_cloud_fraction"
CLOUD = "cloud"
NO_TRANSPARENCY = "no_transparency"
TRANSPARENCY = "transparency"
NO_SITE = "no_site"
SUN = "sun"
MOON = "moon"
NO_TEMPERATURE = "no_temperature"
NO_DARK = "no_dark"
DETECTION = "detection"
SKY_LEVEL = "sky_level"


class SkyFlatError(Exception):
    """The night sky cannot give a flat. The message never names a path."""


# --- The options ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """The settings of the sky builder.

    `bin_factor` and `high_pass_px` (binned pixels) set the average and the split of its parts.
    `polaris_mask_px` is the radius, in pixels of the frame, of the disk that hides Polaris and its
    halo, and `center_xy` (sensor pixels) moves the optical center from the middle of the frame.
    The selection thresholds are `max_sun_elevation_deg`, `max_moon_illumination` (the Moon counts
    only above `moon_min_elevation_deg`), `max_cloud_fraction`, `min_transparency`, `sky_tolerance`
    (a fraction of the median sky level), and `min_exposure_s`. A frame that lacks the cloud
    fraction, the transparency, or the site stays out unless `accept_unchecked` is true.
    `min_frames` and `min_roll_deg` set the warnings, and `ring_limit` the bump (a fraction) that
    says that the mask of Polaris is too small. `radial_limit` is the change of the radial profile
    (a fraction) that a base flat takes from the sky.
    """

    bin_factor: int = 4
    high_pass_px: float = 40.0
    polaris_mask_px: float = 400.0
    center_xy: tuple[float, float] | None = None
    min_frames: int = 20
    min_roll_deg: float = 60.0
    max_sun_elevation_deg: float = -18.0
    max_moon_illumination: float = 0.25
    moon_min_elevation_deg: float = 0.0
    max_cloud_fraction: float = 0.1
    min_transparency: float = 0.95
    sky_tolerance: float = 0.10
    min_exposure_s: float = 5.0
    accept_unchecked: bool = False
    ring_limit: float = 0.003
    edge_margin_px: float = flat_report.EDGE_MARGIN_PX
    radial_limit: float = flat_base.RADIAL_LIMIT

    def __post_init__(self) -> None:
        if self.bin_factor < 1 or self.high_pass_px <= 0 or self.polaris_mask_px < 0:
            raise ValueError("the binning, the high-pass width, and the mask must be positive")
        if not 0 < self.sky_tolerance < 1 or not 0 <= self.max_cloud_fraction <= 1:
            raise ValueError("the sky tolerance and the cloud fraction must lie between 0 and 1")
        if self.min_frames < 1 or self.min_roll_deg < 0:
            raise ValueError("the warning limits must not be negative")
        if not 0 < self.radial_limit < 1:
            raise ValueError("the radial limit must lie between 0 and 1")


# --- The roll -------------------------------------------------------------------------------


def roll_deg(t_utc_ns: int) -> float:
    """The roll of the sky in the sensor frame at a time: the Earth rotation angle, in degrees."""
    from seeingmon.survey.apparent import earth_rotation_angle

    return math.degrees(earth_rotation_angle(t_utc_ns))


def roll_coverage(rolls: Sequence[float]) -> float:
    """The angle that rolls cover about the pole: 360 degrees minus the largest gap between them."""
    if len(rolls) < 2:
        return 0.0
    ordered = np.sort(np.mod(np.asarray(rolls, dtype=np.float64), 360.0))
    gaps = np.diff(np.concatenate([ordered, [ordered[0] + 360.0]]))
    return float(360.0 - gaps.max())


def polaris_separation_deg(t_utc_ns: int) -> float:
    """The angle between Polaris and the pole of date, in degrees, at a time."""
    from seeingmon.survey import apparent

    epoch = apparent.epoch_from_utc_ns(t_utc_ns)
    vector = apparent.apparent_vectors_for(apparent.POLARIS, epoch)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(vector[2])))))


def fit_circle(x: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    """The circle that fits points best (an algebraic fit): its center `(x, y)` and its radius."""
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    target = x * x + y * y
    (cx, cy, c), *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(cx), float(cy), float(math.sqrt(max(c + cx * cx + cy * cy, 0.0)))


# --- The accumulator ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """What the accumulator remembers of a frame that it added.

    `sky_rate` is the sky level above the dark in counts per second per pixel, and `polaris_x` and
    `polaris_y` are the sensor position of Polaris, or `nan` when the frame had no clear one.
    """

    t_utc_ns: int
    roll_deg: float
    sky_rate: float
    polaris_x: float
    polaris_y: float


@dataclass(slots=True)
class Accumulator:
    """The running sums of the masked, normalized frames at `bin_factor` x `bin_factor` binning."""

    shape: tuple[int, int]  # the sensor, as (height, width)
    bin_factor: int
    mode: str
    sum: FloatArray
    sumsq: FloatArray
    count: IntArray
    frames: list[FrameRecord] = field(default_factory=list)
    settings: dict[str, float] = field(default_factory=dict)

    @classmethod
    def empty(
        cls, shape: tuple[int, int], bin_factor: int, mode: str, settings: dict[str, float]
    ) -> Accumulator:
        binned = (shape[0] // bin_factor, shape[1] // bin_factor)
        return cls(
            shape,
            bin_factor,
            mode,
            np.zeros(binned),
            np.zeros(binned),
            np.zeros(binned, dtype=np.int64),
            [],
            dict(settings),
        )

    def times(self) -> set[int]:
        return {record.t_utc_ns for record in self.frames}

    def add(self, part: Contribution, record: FrameRecord) -> None:
        self.sum += part.sum
        self.sumsq += part.sumsq
        self.count += part.count
        self.frames.append(record)

    def save(self, path: Path) -> None:
        """Write the accumulator atomically: a temporary file in the folder, then a rename."""
        from seeingmon.store.layout import write_atomic

        meta = {
            "format": ACCUMULATOR_FORMAT,
            "shape": list(self.shape),
            "bin_factor": self.bin_factor,
            "mode": self.mode,
            "settings": self.settings,
        }
        order = np.argsort([r.t_utc_ns for r in self.frames], kind="stable")
        frames = [self.frames[i] for i in order]
        arrays: dict[str, Any] = {
            "meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            "sum": self.sum,
            "sumsq": self.sumsq,
            "count": self.count,
            "t_utc_ns": np.array([r.t_utc_ns for r in frames], dtype=np.int64),
            "roll_deg": np.array([r.roll_deg for r in frames], dtype=np.float64),
            "sky_rate": np.array([r.sky_rate for r in frames], dtype=np.float64),
            "polaris_x": np.array([r.polaris_x for r in frames], dtype=np.float64),
            "polaris_y": np.array([r.polaris_y for r in frames], dtype=np.float64),
        }

        def write(handle: Any) -> None:
            np.savez_compressed(handle, **arrays)

        write_atomic(path, write)

    @classmethod
    def load(cls, path: Path) -> Accumulator:
        """Read an accumulator file. A file that is not one raises `SkyFlatError`."""
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(bytes(data["meta"]).decode("utf-8"))
                if meta.get("format") != ACCUMULATOR_FORMAT:
                    raise SkyFlatError("the accumulator file has an unknown format")
                acc = cls(
                    shape=(int(meta["shape"][0]), int(meta["shape"][1])),
                    bin_factor=int(meta["bin_factor"]),
                    mode=str(meta["mode"]),
                    sum=np.array(data["sum"], dtype=np.float64),
                    sumsq=np.array(data["sumsq"], dtype=np.float64),
                    count=np.array(data["count"], dtype=np.int64),
                    settings={str(k): float(v) for k, v in dict(meta["settings"]).items()},
                )
                columns = [
                    np.asarray(data[name])
                    for name in ("t_utc_ns", "roll_deg", "sky_rate", "polaris_x", "polaris_y")
                ]
        except SkyFlatError:
            raise
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError) as error:
            raise SkyFlatError(
                f"cannot read the accumulator file: {type(error).__name__}"
            ) from None
        expected = (acc.shape[0] // acc.bin_factor, acc.shape[1] // acc.bin_factor)
        if acc.sum.shape != expected or acc.count.shape != expected:
            raise SkyFlatError("the sums of the accumulator do not fit its size")
        acc.frames = [
            FrameRecord(int(t), float(r), float(s), float(px), float(py))
            for t, r, s, px, py in zip(*columns, strict=True)
        ]
        return acc


# --- Selection ------------------------------------------------------------------------------


def sun_moon_reason(t_utc_ns: int, site: SiteConfig | None, options: BuildOptions) -> str | None:
    """Why the Sun or the Moon rules a frame out, or `None`. `no_site` when the site is unknown."""
    if site is None:
        return NO_SITE
    from seeingmon.scheduler.ephemeris import sun_elevation_deg
    from seeingmon.services.core.moon import moon_elevation_deg, moon_illumination

    if sun_elevation_deg(t_utc_ns, site.latitude_deg, site.longitude_deg) > (
        options.max_sun_elevation_deg
    ):
        return SUN
    elevation = moon_elevation_deg(t_utc_ns, site.latitude_deg, site.longitude_deg)
    if elevation > options.moon_min_elevation_deg and moon_illumination(t_utc_ns) > (
        options.max_moon_illumination
    ):
        return MOON
    return None


def header_reason(
    meta: FrameMeta, mode: str, site: SiteConfig | None, options: BuildOptions
) -> tuple[str | None, list[str]]:
    """The reason that the header rules a frame out, and the tests that it could not run.

    The second value lists the reasons that `accept_unchecked` let pass.
    """
    unchecked: list[str] = []
    if meta.t_utc_ns is None:
        return NO_TIME, unchecked
    if meta.time_invalid:
        return TIME_INVALID, unchecked
    if meta.mode != mode or meta.origin != (0, 0):
        return WRONG_SIZE, unchecked
    if meta.exposure_s is None or meta.exposure_s < options.min_exposure_s:
        return SHORT, unchecked
    if meta.events:
        return EVENT, unchecked
    if meta.cloud_fraction is None:
        if not options.accept_unchecked:
            return NO_CLOUD, unchecked
        unchecked.append(NO_CLOUD)
    elif meta.cloud_fraction >= options.max_cloud_fraction:
        return CLOUD, unchecked
    if meta.transparency is None:
        if not options.accept_unchecked:
            return NO_TRANSPARENCY, unchecked
        unchecked.append(NO_TRANSPARENCY)
    elif meta.transparency < options.min_transparency:
        return TRANSPARENCY, unchecked
    sky = sun_moon_reason(meta.t_utc_ns, site, options)
    if sky == NO_SITE:
        if not options.accept_unchecked:
            return NO_SITE, unchecked
        unchecked.append(NO_SITE)
    elif sky is not None:
        return sky, unchecked
    if meta.temperature_c is None:
        return NO_TEMPERATURE, unchecked
    return None, unchecked


def describe_reasons(counts: dict[str, int], options: BuildOptions) -> str:
    """The reasons as text with their counts, in the order of the tests."""
    texts: dict[str, tuple[str, str]] = {
        UNREADABLE: ("file", "that cannot be read"),
        NO_TIME: ("frame", "without a time"),
        TIME_INVALID: ("frame", "with an unsynchronized clock"),
        WRONG_SIZE: ("frame", "of another readout mode or size"),
        SHORT: ("frame", f"shorter than {options.min_exposure_s:g} s"),
        EVENT: ("event frame", "(no pointing, a moved pointing, clouds, or a bright sky)"),
        NO_CLOUD: ("frame", "without a cloud fraction"),
        CLOUD: ("frame", f"with a cloud fraction of {options.max_cloud_fraction:g} or more"),
        NO_TRANSPARENCY: ("frame", "without a transparency"),
        TRANSPARENCY: ("frame", f"with a transparency under {options.min_transparency:g}"),
        NO_SITE: ("frame", "that the Sun and the Moon could not rule out (no [site])"),
        SUN: ("frame", f"with the Sun above {options.max_sun_elevation_deg:g} degrees"),
        MOON: (
            "frame",
            f"with the Moon up and more than {100 * options.max_moon_illumination:g}% lit",
        ),
        NO_TEMPERATURE: ("frame", "without a sensor temperature"),
        NO_DARK: ("frame", "that no dark set of the library covers"),
        DETECTION: ("frame", "where star detection failed"),
        SKY_LEVEL: (
            "frame",
            f"with a sky level more than {100 * options.sky_tolerance:g}% from the median",
        ),
    }
    parts: list[str] = []
    for key, (noun, rest) in texts.items():
        number = counts.get(key, 0)
        if number > 0:
            parts.append(f"{number} {noun}{'' if number == 1 else 's'} {rest}")
    return ", ".join(parts)


# --- The dark and the hot pixels ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DarkChoice:
    """The dark of one frame: the model level, and a master image when a set matches."""

    level_dn: float
    master: Float32Array | None
    set_name: str | None
    set_dark_dn: float


class DarkSource:
    """The dark library as the sky builder uses it: models, matching sets, and hot pixels.

    The level comes from the dark model of the readout mode and the gain, as in the survey
    pipeline. A set whose temperature lies within `tolerance_c` of the frame, and whose exposure is
    within 5%, also gives a per-pixel master, shifted to the level of the model. The class keeps one
    master in memory at a time.
    """

    def __init__(
        self,
        library: DarkLibrary | None,
        *,
        doubling_c: float,
        tolerance_c: float,
        configured_hot: BoolArray | None = None,
    ) -> None:
        self._library = library
        self._doubling_c = doubling_c
        self._tolerance_c = tolerance_c
        self._configured_hot = configured_hot
        self._models: dict[tuple[str, int], DarkModel | None] = {}
        self._master: tuple[str, Float32Array] | None = None
        self._hot: tuple[str, BoolArray] | None = None
        self.sets_used: Counter[str] = Counter()
        self.scalar_frames = 0

    def model(self, mode: str, gain: int) -> DarkModel | None:
        key = (mode, gain)
        if key not in self._models:
            self._models[key] = (
                None
                if self._library is None
                else self._library.model(mode, gain, prior_doubling_c=self._doubling_c)
            )
        return self._models[key]

    def scalar_level(self, meta: FrameMeta) -> float | None:
        """The level of the dark model for a frame, or `None` without a model."""
        if meta.mode is None or meta.gain is None or meta.temperature_c is None:
            return None
        model = self.model(meta.mode, meta.gain)
        if model is None or meta.exposure_s is None:
            return None
        return float(model.level_dn(meta.temperature_c, meta.exposure_s))

    def choose(self, meta: FrameMeta, shape: tuple[int, int]) -> DarkChoice | None:
        """The dark for a frame, or `None` when the library has no model for its mode and gain."""
        level = self.scalar_level(meta)
        if level is None:
            return None
        assert self._library is not None
        assert meta.mode is not None
        assert meta.gain is not None
        assert meta.temperature_c is not None
        assert meta.exposure_s is not None
        chosen = self._library.nearest(meta.mode, meta.gain, meta.temperature_c)
        if (
            chosen is not None
            and abs(chosen.temperature_c - meta.temperature_c) <= self._tolerance_c
            and abs(chosen.exposure_s - meta.exposure_s)
            <= DARK_EXPOSURE_TOLERANCE * meta.exposure_s
            and (chosen.height_px, chosen.width_px) == shape
        ):
            return DarkChoice(level, self._load_master(chosen), chosen.name, chosen.dark_dn)
        return DarkChoice(level, None, None, 0.0)

    def _load_master(self, chosen: DarkSet) -> Float32Array:
        assert self._library is not None
        if self._master is None or self._master[0] != chosen.name:
            master = self._library.load_master(chosen).astype(np.float32)
            self._master = (chosen.name, master)
        return self._master[1]

    def hot_mask(self, meta: FrameMeta, shape: tuple[int, int]) -> BoolArray | None:
        """The hot pixels of the configured mask and of the library set nearest the temperature."""
        masks: list[BoolArray] = []
        if self._configured_hot is not None and self._configured_hot.shape == shape:
            masks.append(self._configured_hot)
        if (
            self._library is not None
            and meta.mode is not None
            and meta.gain is not None
            and meta.temperature_c is not None
        ):
            chosen = self._library.nearest(meta.mode, meta.gain, meta.temperature_c)
            if chosen is not None and (chosen.height_px, chosen.width_px) == shape:
                if self._hot is None or self._hot[0] != chosen.name:
                    mask = np.zeros(shape, dtype=np.bool_)
                    columns, rows, _ = self._library.hot_pixels(chosen)
                    mask[rows, columns] = True
                    self._hot = (chosen.name, mask)
                masks.append(self._hot[1])
        if not masks:
            return None
        return masks[0] if len(masks) == 1 else np.asarray(masks[0] | masks[1], dtype=np.bool_)

    def record(self, choice: DarkChoice) -> None:
        if choice.set_name is None:
            self.scalar_frames += 1
        else:
            self.sets_used[choice.set_name] += 1

    def describe(self) -> str:
        """The dark that the used frames had, for the report."""
        if self._library is None:
            return "none"
        parts: list[str] = []
        for name, count in sorted(self.sets_used.items()):
            parts.append(f"the master dark of {name_to_text(name)} for {count} frames")
        if self.scalar_frames:
            parts.append(f"the level of the dark model alone for {self.scalar_frames} frames")
        return "; ".join(parts) if parts else "no frame used a dark"


def name_to_text(name: str) -> str:
    """A dark set name, such as `dark-20261001T120000Z-bin2-g120.fits`, as a date."""
    stamp = name.split("-")[1] if name.count("-") >= 2 else ""
    if len(stamp) >= 8 and stamp[:8].isdigit():
        return f"the set of {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    return "a set"


# --- Masks ----------------------------------------------------------------------------------


def paint_disks(mask: BoolArray, xs: FloatArray, ys: FloatArray, radii: FloatArray) -> None:
    """Set the pixels of a disk around each position, in place. `mask` is `(height, width)`."""
    height, width = mask.shape
    for x, y, radius in zip(xs, ys, radii, strict=True):
        reach = math.ceil(radius)
        x0, x1 = max(round(x) - reach, 0), min(round(x) + reach + 1, width)
        y0, y1 = max(round(y) - reach, 0), min(round(y) + reach + 1, height)
        if x0 >= x1 or y0 >= y1:
            continue
        gy, gx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (gx - x) ** 2 + (gy - y) ** 2 <= radius * radius


def star_radii(detections: Detections, sky_px: float) -> FloatArray:
    """The radius of the mask of each star: its size and trail, and a wing that grows with its flux.

    A saturated star gets at least the radius that the detector's own rule gives
    (`seeingmon.survey.detect.star_mask`).
    """
    from seeingmon.survey.centroid import FWHM_PER_SIGMA
    from seeingmon.survey.detect import StarFlag

    sigma = detections.fwhm_px / FWHM_PER_SIGMA
    radius = np.maximum(3.0 * sigma + detections.trail_length_px / 2.0, 3.0)
    saturated = detections.has(StarFlag.SATURATED)
    radius = np.where(
        saturated,
        np.maximum(radius, 2.0 * np.sqrt(np.maximum(detections.n_pixels, 1))),
        radius,
    )
    wing = HALO_PX_PER_CUBE_ROOT * np.cbrt(np.maximum(detections.flux, 0.0) / max(sky_px, 1.0))
    return np.asarray(radius + np.minimum(wing, MAX_HALO_PX), dtype=np.float64)


def find_polaris(detections: Detections) -> int | None:
    """The index of Polaris among the detections: the brightest star, if it stands well out."""
    count = len(detections)
    if count == 0:
        return None
    order = np.argsort(-detections.flux)
    if count > 1 and detections.flux[order[0]] < POLARIS_FLUX_RATIO * detections.flux[order[1]]:
        return None
    return int(order[0])


def edge_mask(shape: tuple[int, int], width: int = EDGE_PX) -> BoolArray:
    mask = np.zeros(shape, dtype=np.bool_)
    if width > 0:
        mask[:width, :] = True
        mask[-width:, :] = True
        mask[:, :width] = True
        mask[:, -width:] = True
    return mask


def grow(mask: BoolArray, pixels: int = 1) -> BoolArray:
    """The mask grown by `pixels` in every direction: each pixel becomes a square around it."""
    grown = mask.copy()
    for _ in range(pixels):
        grown[1:, :] |= grown[:-1, :]
        grown[:-1, :] |= grown[1:, :]
        grown[:, 1:] |= grown[:, :-1]
        grown[:, :-1] |= grown[:, 1:]
    return grown


# --- One frame ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Contribution:
    """What one frame adds: the sums at the binned size, its sky level, and where Polaris is."""

    sum: FloatArray
    sumsq: FloatArray
    count: IntArray
    level_dn: float
    polaris: tuple[float, float] | None
    n_stars: int


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Everything that processing a frame needs beside the frame."""

    profile: Profile
    survey: SurveyConfig
    options: BuildOptions
    detect: DetectOptions
    darks: DarkSource


def detect_options(survey: SurveyConfig) -> DetectOptions:
    """The detector settings of the survey configuration."""
    from seeingmon.survey.detect import DetectOptions

    return DetectOptions.from_config(survey.detect)


def process_frame(
    pixels: npt.NDArray[Any], meta: FrameMeta, choice: DarkChoice, context: FrameContext
) -> Contribution | None:
    """Mask a frame, normalize it by its own sky level, and bin it. `None`: the frame is unusable.

    `pixels` are native counts. The frame is unusable when star detection fails or too few pixels
    remain to measure the sky.
    """
    from seeingmon.survey.detect import DetectionError, detect_stars
    from seeingmon.survey.sky import SkyOptions, measure_sky

    profile, options = context.profile, context.options
    assert meta.mode is not None
    assert meta.gain is not None
    assert meta.exposure_s is not None
    readout = profile.mode(meta.mode)
    saturation = profile.saturation(meta.mode, meta.gain).native_dn
    e_per_adu = profile.e_per_adu(meta.mode, meta.gain)
    scale = profile.plate_scale_arcsec_per_px(readout)
    shape = (int(pixels.shape[0]), int(pixels.shape[1]))
    native = np.asarray(pixels, dtype=np.float32)
    hot = context.darks.hot_mask(meta, shape)
    try:
        detections = detect_stars(
            native,
            saturation_dn=saturation,
            options=context.detect,
            e_per_adu=e_per_adu,
            hot_pixels=hot,
        )
    except DetectionError:
        return None

    diff = native - choice.master if choice.master is not None else native.copy()
    diff -= np.float32(
        choice.level_dn - choice.set_dark_dn if choice.master is not None else choice.level_dn
    )
    rough = float(np.median(diff[::PREFLIGHT_STRIDE, ::PREFLIGHT_STRIDE]))
    mask = np.zeros(shape, dtype=np.bool_)
    radii = star_radii(detections, max(rough, 1.0))
    polaris = find_polaris(detections)
    if polaris is not None:
        radii[polaris] = options.polaris_mask_px  # the option, whatever the detector says
    paint_disks(mask, detections.x, detections.y, radii)
    mask |= grow(native >= np.float32(SATURATION_FRACTION * saturation))
    if hot is not None:
        mask |= hot
    mask |= edge_mask(shape)
    measured = measure_sky(
        diff,
        star_mask=mask,
        dark_level_dn=0.0,
        flat=None,
        exposure_s=meta.exposure_s,
        e_per_adu=e_per_adu,
        scale_arcsec_px=scale,
        saturation_dn=float("inf"),
        options=SkyOptions(edge_px=0),
    )
    if measured is None or measured.level_dn <= 0:
        return None
    valid = ~mask
    values = np.where(valid, diff / np.float32(measured.level_dn), np.float32(0.0))
    factor = options.bin_factor
    total = flat_report.block_sum(values, factor)
    values *= values
    squares = flat_report.block_sum(values, factor)
    counts = flat_report.block_sum(valid, factor).astype(np.int64)
    fix = None if polaris is None else (float(detections.x[polaris]), float(detections.y[polaris]))
    return Contribution(total, squares, counts, measured.level_dn, fix, len(detections))


def preflight_rate(pixels: npt.NDArray[Any], meta: FrameMeta, level_dn: float) -> float:
    """A cheap sky rate (counts per second per pixel above the dark) from every 8th pixel."""
    assert meta.exposure_s is not None
    sample = pixels[::PREFLIGHT_STRIDE, ::PREFLIGHT_STRIDE]
    return (float(np.median(sample)) - level_dn) / meta.exposure_s


# --- The average ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SkyAverage:
    """The mean of the accumulated frames in the sensor frame, scaled to a median of 1."""

    mean: FloatArray
    valid: BoolArray
    count: IntArray
    noise_frame: float  # the scatter of one pixel of one frame, as a fraction
    noise_mean: float  # the noise of the mean in a binned pixel, as a fraction
    noise_scale: FloatArray  # the noise of each binned pixel over the typical one (1 or more)


def average_of(acc: Accumulator) -> SkyAverage:
    """The mean `sum / count`, with the pixels that few frames covered marked invalid."""
    if not acc.frames:
        raise SkyFlatError("the accumulator holds no frame")
    counted = acc.count > 0
    median_count = float(np.median(acc.count[counted])) if counted.any() else 0.0
    valid = acc.count >= max(1.0, 0.2 * median_count)
    mean = np.divide(acc.sum, acc.count, out=np.ones_like(acc.sum), where=counted)
    reference = float(np.median(mean[valid]))
    if reference <= 0:
        raise SkyFlatError("the mean sky is not positive")
    scale = 1.0 / reference
    mean = np.where(valid, mean * scale, 1.0)
    variance = np.divide(acc.sumsq, acc.count, out=np.zeros_like(acc.sum), where=counted) - (
        np.divide(acc.sum, acc.count, out=np.zeros_like(acc.sum), where=counted) ** 2
    )
    variance = np.maximum(variance, 0.0) * scale * scale
    noise_frame = float(math.sqrt(float(np.median(variance[valid]))))
    per_bin = np.sqrt(variance[valid] / acc.count[valid])
    typical = float(np.median(acc.count[valid]))
    ratio = np.sqrt(typical / np.maximum(acc.count, 1))
    noise_scale = np.where(valid, np.maximum(ratio, 1.0), 1.0)
    return SkyAverage(mean, valid, acc.count, noise_frame, float(np.median(per_bin)), noise_scale)


@dataclass(frozen=True, slots=True)
class RingCheck:
    """The bump of the mean sky at the radius of the orbit of Polaris."""

    radius_px: float
    center_xy: tuple[float, float]
    bump: float | None
    measured: bool  # the circle comes from the positions of Polaris and not from the ephemeris


def ring_check(
    average: SkyAverage,
    acc: Accumulator,
    *,
    optical_center: tuple[float, float],
    scale_arcsec_px: float,
) -> RingCheck:
    """Compare the mean sky at the orbit of Polaris with a smooth baseline fitted outside it.

    The circle comes from the sensor positions of Polaris in the frames when enough of them
    spread over 90 degrees (the pole is the center, and the orbit its radius), and from the
    ephemeris and the optical center otherwise.
    """
    factor = acc.bin_factor
    times = [record.t_utc_ns for record in acc.frames]
    expected_px = polaris_separation_deg(int(np.median(times))) * 3600.0 / scale_arcsec_px
    center, radius_px, measured = optical_center, expected_px, False
    fixes = np.array(
        [(r.polaris_x, r.polaris_y) for r in acc.frames if np.isfinite(r.polaris_x)],
        dtype=np.float64,
    )
    if len(fixes) >= MIN_POLARIS_FIXES:
        cx, cy, radius = fit_circle(fixes[:, 0], fixes[:, 1])
        angles = np.degrees(np.arctan2(fixes[:, 1] - cy, fixes[:, 0] - cx))
        if roll_coverage(list(angles)) >= MIN_POLARIS_SPREAD_DEG and (
            0.5 * expected_px <= radius <= 2.0 * expected_px
        ):
            center, radius_px, measured = (cx, cy), radius, True
    center_b = flat_report.binned_position(center, factor)
    radius_b = radius_px / factor
    return RingCheck(
        radius_px=radius_px,
        center_xy=center,
        bump=_bump(average, center_b, radius_b),
        measured=measured,
    )


def _bump(average: SkyAverage, center_b: tuple[float, float], radius_b: float) -> float | None:
    """The mean of the profile at the orbit over a baseline fitted to the rings on both sides.

    The baseline is a parabola in the radius, fitted to the rings between 0.2 and 0.7 orbit radii
    from the orbit (inside and outside), so it follows the vignetting near the orbit and leaves
    the ring out.
    """
    radius = flat_report.radius_map(average.mean.shape, center_b)
    profile = flat_report.azimuthal_profile(average.mean, radius, valid=average.valid)
    r, value, weight = profile.radius, profile.value, profile.count
    gap = np.abs(r - radius_b)
    flank = (gap >= max(0.2 * radius_b, 6.0)) & (gap <= max(0.7 * radius_b, 20.0))
    inside = gap < max(0.08 * radius_b, 2.5)
    if int(flank.sum()) < 24 or int(inside.sum()) < 2:
        return None
    coefficients = np.polyfit(r[flank], value[flank], 2, w=np.sqrt(weight[flank]))
    baseline = np.polyval(coefficients, r[inside])
    return float(np.mean(value[inside] / baseline - 1.0))


@dataclass(frozen=True, slots=True)
class SkyFlat:
    """The flat of the sky average and its parts, all at the binned size."""

    flat_binned: FloatArray
    decomposition: flat_report.Decomposition
    center_binned: tuple[float, float]


def flat_from_average(
    average: SkyAverage, *, center_binned: tuple[float, float], high_pass_px: float
) -> SkyFlat:
    """The flat of a sky average: its radial part times its fine part, with a median of 1."""
    parts = flat_report.decompose(
        average.mean, center_xy=center_binned, high_pass_px=high_pass_px, valid=average.valid
    )
    flat = parts.radial_map * parts.fine
    flat = flat / float(np.median(flat))
    return SkyFlat(flat, parts, center_binned)


# --- The run --------------------------------------------------------------------------------


@dataclass(slots=True)
class Selection:
    """The frames of a folder after the header tests, and the counts of the ones left out."""

    candidates: list[tuple[Path, FrameMeta]] = field(default_factory=list)
    rejected: Counter[str] = field(default_factory=Counter)
    unchecked: Counter[str] = field(default_factory=Counter)
    found: int = 0
    already: int = 0


def list_frame_files(folder: Path) -> list[Path]:
    """The FITS files under a folder, in name order. Hidden files (temporary writes) stay out."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in FITS_SUFFIXES and not path.name.startswith(".")
    )


def screen_headers(
    files: Sequence[Path],
    *,
    mode: str,
    known: set[int],
    site: SiteConfig | None,
    options: BuildOptions,
) -> Selection:
    """Read the header of every file and apply the tests that need no pixels."""
    selection = Selection()
    seen: set[int] = set()
    for path in files:
        selection.found += 1
        try:
            meta = parse_frame_header(read_frame_header(path))
        except (OSError, ValueError):
            selection.rejected[UNREADABLE] += 1
            continue
        if meta.t_utc_ns is not None and (meta.t_utc_ns in known or meta.t_utc_ns in seen):
            selection.already += 1
            continue
        reason, unchecked = header_reason(meta, mode, site, options)
        if reason is not None:
            selection.rejected[reason] += 1
            continue
        assert meta.t_utc_ns is not None
        seen.add(meta.t_utc_ns)
        selection.unchecked.update(unchecked)
        selection.candidates.append((path, meta))
    return selection


@dataclass(frozen=True, slots=True)
class SkyResult:
    """The flat of a run and what the report says about it.

    `flat` is the flat that the command writes: the flat of the sky alone, or the updated base
    flat when `updated` is true. With a base flat and no update, the command writes nothing.
    `base` holds the comparison with the base flat, and `applied` what an update applies (or
    would apply).
    """

    flat: Float32Array
    options: BuildOptions
    selection: Selection
    used_now: int
    accumulator: Accumulator
    average: SkyAverage
    summary: flat_report.FlatSummary
    ring: RingCheck
    sky_flat: SkyFlat
    dark_note: str
    warnings: tuple[str, ...]
    elapsed_s: float | None
    base: flat_base.BaseComparison | None = None
    applied: flat_base.Applied | None = None
    updated: bool = False


ProgressFn = Callable[[str], None]


def build_sky_flat(
    folder: Path | None,
    *,
    profile: Profile,
    survey: SurveyConfig,
    library: DarkLibrary | None,
    site: SiteConfig | None,
    hot_mask: BoolArray | None = None,
    accumulator_path: Path | None = None,
    options: BuildOptions | None = None,
    clock: Clock | None = None,
    progress: ProgressFn | None = None,
    base_flat: Float32Array | None = None,
    update: bool = False,
) -> SkyResult:
    """Add the new frames of a folder to the accumulator and build the flat. See the module text.

    `base_flat` is a flat of the whole sensor (a median of 1), and the sky is compared with it.
    With `update`, the result is the base flat with the changes that the sky shows.
    """
    cfg = options or BuildOptions()
    started = None if clock is None else clock.monotonic_ns()
    say = progress or (lambda _: None)
    readout = profile.survey_readout
    mode = profile.survey_mode.mode
    shape = (readout.height_px, readout.width_px)
    if update and base_flat is None:
        raise SkyFlatError("an update needs a base flat")
    if base_flat is not None and base_flat.shape != shape:
        raise SkyFlatError(
            f"the base flat has {base_flat.shape[1]} x {base_flat.shape[0]} pixels, and the "
            f"survey mode has {shape[1]} x {shape[0]}: use a flat of this mode"
        )
    center = cfg.center_xy or ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    if not (0.0 <= center[0] <= shape[1] - 1 and 0.0 <= center[1] <= shape[0] - 1):
        raise SkyFlatError(
            f"the optical center ({center[0]:g}, {center[1]:g}) lies off the frame of "
            f"{shape[1]} x {shape[0]} pixels: give --center-x and --center-y on the frame"
        )
    settings = {"polaris_mask_px": cfg.polaris_mask_px, "edge_px": float(EDGE_PX)}
    if accumulator_path is not None and accumulator_path.exists():
        acc = Accumulator.load(accumulator_path)
        if (acc.shape, acc.bin_factor, acc.mode) != (shape, cfg.bin_factor, mode):
            raise SkyFlatError(
                f"the accumulator holds {acc.bin_factor} x {acc.bin_factor} binned sums of "
                f"{acc.shape[1]} x {acc.shape[0]} frames of {acc.mode}, and this run needs "
                f"{cfg.bin_factor} x {cfg.bin_factor} binned sums of {shape[1]} x {shape[0]} "
                f"frames of {mode}: use another accumulator file"
            )
    else:
        acc = Accumulator.empty(shape, cfg.bin_factor, mode, settings)
    warnings: list[str] = []
    old_mask = acc.settings.get("polaris_mask_px")
    if acc.frames and old_mask is not None and old_mask != cfg.polaris_mask_px:
        acc.settings["mixed_masks"] = 1.0
    if acc.settings.get("mixed_masks"):
        warnings.append(
            "The accumulator mixes frames that hid Polaris with different radii (the first with "
            f"{acc.settings.get('polaris_mask_px', cfg.polaris_mask_px):g} px, and this run with "
            f"{cfg.polaris_mask_px:g} px). Delete the accumulator and build again to use one."
        )

    files = [] if folder is None else list_frame_files(folder)
    selection = screen_headers(files, mode=mode, known=acc.times(), site=site, options=cfg)
    darks = DarkSource(
        library,
        doubling_c=survey.dark.doubling_c,
        tolerance_c=survey.dark.temperature_tolerance_c,
        configured_hot=hot_mask,
    )
    context = FrameContext(profile, survey, cfg, detect_options(survey), darks)
    candidates = screen_sky_levels(selection, acc, darks, shape, cfg)
    added = 0
    for number, (path, meta, _rate) in enumerate(candidates, start=1):
        tick = None if clock is None else clock.monotonic_ns()
        assert meta.t_utc_ns is not None
        choice = darks.choose(meta, shape)
        if choice is None:  # the library lost the set between the screening and now
            selection.rejected[NO_DARK] += 1
            continue
        try:
            pixels = read_frame_fits(path).pixels
        except (OSError, ValueError):
            selection.rejected[UNREADABLE] += 1
            continue
        part = process_frame(pixels, meta, choice, context)
        if part is None:
            selection.rejected[DETECTION] += 1
            continue
        assert meta.exposure_s is not None
        acc.add(
            part,
            FrameRecord(
                t_utc_ns=meta.t_utc_ns,
                roll_deg=roll_deg(meta.t_utc_ns),
                sky_rate=part.level_dn / meta.exposure_s,
                polaris_x=math.nan if part.polaris is None else part.polaris[0],
                polaris_y=math.nan if part.polaris is None else part.polaris[1],
            ),
        )
        darks.record(choice)
        added += 1
        if clock is not None and tick is not None:
            say(
                f"frame {number} of {len(candidates)}: "
                f"{(clock.monotonic_ns() - tick) / NS_PER_S:.1f} s"
            )
    if not acc.frames:
        raise SkyFlatError(_nothing_to_use(selection, cfg))
    if accumulator_path is not None and (added or not accumulator_path.exists()):
        try:
            acc.save(accumulator_path)
        except OSError as error:
            raise SkyFlatError(
                f"cannot write the accumulator file: {error.strerror or type(error).__name__}"
            ) from None
    average = average_of(acc)
    scale = profile.plate_scale_arcsec_per_px(readout)
    center_binned = flat_report.binned_position(center, cfg.bin_factor)
    sky_flat = flat_from_average(
        average, center_binned=center_binned, high_pass_px=cfg.high_pass_px
    )
    summary = flat_report.summarize_flat(
        sky_flat.flat_binned,
        factor=cfg.bin_factor,
        sensor_shape=shape,
        scale_arcsec_px=scale,
        center_xy=center_binned,
        high_pass_px=cfg.high_pass_px,
        edge_margin_px=cfg.edge_margin_px,
        noise_scale=average.noise_scale,
        valid=average.valid,
    )
    ring = ring_check(average, acc, optical_center=center, scale_arcsec_px=scale)
    ring_ok = ring.bump is None or ring.bump <= cfg.ring_limit
    comparison: flat_base.BaseComparison | None = None
    plan: flat_base.Applied | None = None
    flat: Float32Array | None = None
    if base_flat is not None:
        comparison = flat_base.compare_with_base(
            average.mean,
            flat_report.block_mean(base_flat, cfg.bin_factor),
            valid=average.valid,
            noise_scale=average.noise_scale,
            factor=cfg.bin_factor,
            sensor_shape=shape,
            scale_arcsec_px=scale,
            center_xy=center_binned,
            high_pass_px=cfg.high_pass_px,
            radial_limit=cfg.radial_limit,
            edge_margin_px=cfg.edge_margin_px,
        )
        plan = flat_base.plan_update(comparison, trusted=ring_ok)
        if update:
            flat = flat_base.apply_update(base_flat, comparison, plan, factor=cfg.bin_factor)
    if flat is None:
        flat = flat_report.upsample_bilinear(sky_flat.flat_binned, cfg.bin_factor, shape)
    rolls = [record.roll_deg for record in acc.frames]
    coverage = roll_coverage(rolls)
    if len(acc.frames) < cfg.min_frames:
        warnings.append(
            f"Only {len(acc.frames)} frames went in, under {cfg.min_frames}. The noise of the fine "
            "structure is high, and the rotation has not averaged the structure of the sky."
        )
    if coverage < cfg.min_roll_deg:
        warnings.append(
            f"The frames cover {coverage:.0f} degrees of roll, under {cfg.min_roll_deg:g}. The "
            "rotation has not averaged the structure of the sky, so the flat holds part of it."
        )
    if not ring_ok and ring.bump is not None:
        warnings.append(
            f"The mean sky has a bump of {100 * ring.bump:.2f}% at the radius of the orbit of "
            f"Polaris, over {100 * cfg.ring_limit:.1f}%: the mask around Polaris is too small. "
            "Raise --polaris-mask-px, and build again from the frames."
        )
    if darks.scalar_frames and library is not None and darks.sets_used:
        warnings.append(
            f"{darks.scalar_frames} frames had no dark set within the temperature tolerance, so "
            "only the level of the dark model came off them."
        )
    if library is not None and not darks.sets_used and darks.scalar_frames:
        warnings.append(
            "No dark set of the library matched the temperature and the exposure of a frame, so "
            "only the level of the dark model came off. The pattern of the dark stays in the flat."
        )
    if selection.unchecked:
        names = ", ".join(sorted(selection.unchecked))
        warnings.append(f"Frames went in unchecked ({names}), because you asked for it.")
    elapsed = (
        None if clock is None or started is None else (clock.monotonic_ns() - started) / NS_PER_S
    )
    return SkyResult(
        flat=flat,
        options=cfg,
        selection=selection,
        used_now=added,
        accumulator=acc,
        average=average,
        summary=summary,
        ring=ring,
        sky_flat=sky_flat,
        dark_note=darks.describe(),
        warnings=tuple(warnings),
        elapsed_s=elapsed,
        base=comparison,
        applied=plan,
        updated=update,
    )


def screen_sky_levels(
    selection: Selection,
    acc: Accumulator,
    darks: DarkSource,
    shape: tuple[int, int],
    options: BuildOptions,
) -> list[tuple[Path, FrameMeta, float]]:
    """Read each candidate's pixels once for a cheap sky rate, and keep the frames near the median.

    The median covers the frames of the accumulator too, so a run on a few new frames judges them
    against the whole record. A frame without a dark model, or of the wrong size, drops here.
    """
    measured: list[tuple[Path, FrameMeta, float]] = []
    for path, meta in selection.candidates:
        level = darks.scalar_level(meta)
        if level is None:
            selection.rejected[NO_DARK] += 1
            continue
        try:
            pixels = read_frame_fits(path).pixels
        except (OSError, ValueError):
            selection.rejected[UNREADABLE] += 1
            continue
        if pixels.shape != shape:
            selection.rejected[WRONG_SIZE] += 1
            continue
        measured.append((path, meta, preflight_rate(pixels, meta, level)))
    rates = [rate for _, _, rate in measured] + [r.sky_rate for r in acc.frames]
    if not rates:
        return []
    median = float(np.median(rates))
    kept: list[tuple[Path, FrameMeta, float]] = []
    for path, meta, rate in measured:
        if median <= 0 or rate <= 0 or abs(rate / median - 1.0) > options.sky_tolerance:
            selection.rejected[SKY_LEVEL] += 1
        else:
            kept.append((path, meta, rate))
    return kept


def _nothing_to_use(selection: Selection, options: BuildOptions) -> str:
    reasons = describe_reasons(dict(selection.rejected), options)
    if selection.found == 0:
        return "the folder holds no FITS file of survey frames"
    text = f"no frame is usable ({selection.found} found"
    if selection.already:
        text += f", {selection.already} already in the accumulator"
    if reasons:
        text += f": {reasons}"
    text += ")"
    if selection.rejected.get(NO_SITE):
        text += ". Set [site] in the configuration, or pass --accept-unchecked"
    elif selection.rejected.get(NO_CLOUD) or selection.rejected.get(NO_TRANSPARENCY):
        text += ". The files lack the cloud fraction or the transparency: pass --accept-unchecked"
    return text


# --- The report -----------------------------------------------------------------------------


def format_sky_report(
    result: SkyResult, *, name: str | None = None, base_name: str | None = None
) -> list[str]:
    """The plain-text report of a run. It names no path of the machine.

    `name` is the file that the command wrote, and `base_name` the base flat. Both are names
    without a folder.
    """
    cfg, selection, acc = result.options, result.selection, result.accumulator
    lines = [
        "Flat from the night sky."
        if result.base is None
        else "Flat from the night sky, compared with a base flat."
    ]
    rejected = sum(selection.rejected.values())
    lines.append(
        f"Frames: {selection.found} found in the folder, {selection.already} already in the "
        f"accumulator, {rejected} rejected, {result.used_now} added now."
    )
    if rejected:
        lines.append(f"  Rejected: {describe_reasons(dict(selection.rejected), cfg)}.")
    times = [record.t_utc_ns for record in acc.frames]
    first, last = utc_ns_to_iso(min(times), digits=0), utc_ns_to_iso(max(times), digits=0)
    rolls = [record.roll_deg for record in acc.frames]
    lines.append(
        f"Accumulator: {len(acc.frames)} frames from {first[:10]} to {last[:10]}. Roll coverage: "
        f"{roll_coverage(rolls):.0f} degrees."
    )
    if result.used_now:  # a run that added no frame processed none, and used no dark
        lines.append(f"Dark: {result.dark_note}.")
    lines.append(
        f"Noise: {100 * result.average.noise_mean:.2f}% per binned pixel ({cfg.bin_factor} x "
        f"{cfg.bin_factor}) in the mean sky. One pixel of one frame scatters by "
        f"{100 * result.average.noise_frame:.1f}%."
    )
    if result.base is not None:
        lines.extend(
            flat_base.comparison_lines(
                result.base, base_name=base_name, edge_margin_px=cfg.edge_margin_px
            )
        )
    else:
        lines.extend(flat_report.profile_lines(result.summary.profile))
        lines.append(
            "Tilt: not determined. The sky cannot tell a tilt of the flat from a gradient of "
            "the sky itself, so the flat holds none. A tilt of the optics of about 1% across "
            "the frame stays in the frames."
        )
        lines.extend(
            flat_report.shadow_lines(result.summary.shadows, depth=result.summary.shadow_depth)
        )
        lines.extend(
            flat_report.edge_artifact_lines(
                result.summary.edge_artifacts,
                margin_px=cfg.edge_margin_px,
                depth=result.summary.shadow_depth,
            )
        )
    ring = result.ring
    if ring.bump is None:
        lines.append("Polaris orbit: the ring check could not run (the frame holds too little).")
    else:
        origin = "from the positions of Polaris" if ring.measured else "from the ephemeris"
        lines.append(
            f"Polaris orbit: radius {ring.radius_px:.0f} px ({origin}), bump "
            f"{100 * ring.bump:+.2f}% against a limit of {100 * cfg.ring_limit:.1f}%."
        )
    if result.elapsed_s is not None and result.used_now:
        lines.append(
            f"Time: {result.elapsed_s:.1f} s, {result.elapsed_s / result.used_now:.1f} s for each "
            "frame added."
        )
    if result.applied is not None:
        lines.extend(flat_base.update_lines(result.applied, written=result.updated))
    lines.extend(f"Warning: {warning}" for warning in result.warnings)
    if name is not None:
        what = ", the base flat with these changes" if result.updated else ""
        lines.append(f"Wrote {name}{what}. Set flat_file in the [survey] table to its path.")
    elif result.base is not None and not result.updated:
        lines.append("Nothing written. Add --update and --out to write the new flat.")
    return lines
