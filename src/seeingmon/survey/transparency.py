"""Transparency, the cloud flag, and the limiting magnitude.

**Transparency.** Polaris sits at a fixed altitude, so the extinction of the atmosphere cannot
be fitted from stars at different airmasses. The survey compares each frame's zero point with
a reference from the clearest conditions that the station has seen:

    transparency = 10 ** (-0.4 * (ZP_ref - ZP))

A frame as clear as the reference has transparency 1, and a frame that loses 0.75 mag to
clouds has 0.5. The reference is a high quantile (the 95th by default) of the zero points of
the usable frames of a long window (a year by default), so that weeks of haze cannot lower it.
A frame is usable when its fit used enough stars, its scatter was small, and it was not
cloudy. A quantile, not a maximum, keeps one lucky frame from setting the reference. A
station that knows its clear-sky zero point can pin it (`pinned_zero_point`), and the pin
replaces the history.

**The history protocol.** The reference needs past zero points, which live in the store as
`sky_quality` records. `ZeroPointHistory` is the small interface that the store implements
later: `zero_points(since, until)` returns the samples in a time range. `MemoryHistory` is a
plain implementation that the analyzer feeds with every result of the running process, and
that a caller seeds from the store at start-up (`samples_from_records`). The analyzer reads
the history when it submits a frame and passes the reference to the worker as a number, so a
worker process needs no access to the store.

**The provisional zero point.** The reference needs `min_samples` usable zero points (20 by
default), and a young history holds fewer. Until the reference exists, `provisional_zero_point`
stands in for it: the median of the usable zero points of the last `fallback_hours` (6 by
default). It serves a frame that has no zero point of its own, so that the sky brightness does
not drop out after a lost pointing solution or a restart. It never sets a transparency, because
a median of recent frames does not describe the clearest conditions that the transparency
measures against.

**The limiting magnitude** is the magnitude at which the frame detects half of the catalog
stars in its field. The function bins the stars by G, makes the detected fraction fall
steadily with magnitude, and interpolates to 0.5. A clear frame detects more than half of the
stars at the faint end of the catalog (G = 13), so it has no measured limit. Then the noise of
the frame predicts one: the magnitude of a star that a 5-sigma detection needs.

**Clouds.** The cloud fraction is the share of expected stars that detection missed (see
`seeingmon.survey.pipeline`). The `cloud` flag marks a fraction of at least
`cloud_flag_fraction` or a transparency below `transparency_flag`.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.survey.geometry import FloatArray
from seeingmon.survey.nights import night_label

BoolArray = npt.NDArray[np.bool_]
SECONDS_PER_HOUR = 3_600.0
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class ZeroPointSample:
    """One zero point from the past: the frame time, the fit, and how clear the frame was."""

    t_utc_ns: int
    zero_point_mag: float
    rms_mag: float | None = None
    n_stars: int = 0
    cloud_fraction: float | None = None


class ZeroPointHistory(Protocol):
    """Past zero points. The store implements this later."""

    def zero_points(self, since_utc_ns: int, until_utc_ns: int) -> Sequence[ZeroPointSample]:
        """The samples with `since_utc_ns <= t_utc_ns < until_utc_ns`, oldest first."""
        ...


def sample_from_record(record: SkyQualityRecord) -> ZeroPointSample | None:
    """The sample that a `sky_quality` record holds, or `None` when it has no zero point."""
    if record.zero_point_mag is None:
        return None
    return ZeroPointSample(
        t_utc_ns=record.t_utc_ns,
        zero_point_mag=record.zero_point_mag,
        rms_mag=record.zero_point_rms_mag,
        n_stars=record.n_stars_used,
        cloud_fraction=record.cloud_fraction,
    )


def samples_from_records(records: Iterable[SkyQualityRecord]) -> list[ZeroPointSample]:
    """The samples of the records that hold a zero point, in the order given."""
    found = (sample_from_record(record) for record in records)
    return [sample for sample in found if sample is not None]


class MemoryHistory:
    """A `ZeroPointHistory` in memory. It keeps the newest `max_samples` and stays sorted."""

    def __init__(
        self, samples: Iterable[ZeroPointSample] = (), *, max_samples: int = 50_000
    ) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        self._max = max_samples
        self._samples: list[ZeroPointSample] = []
        self._times: list[int] = []
        for sample in samples:
            self.add(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: ZeroPointSample) -> None:
        """Add a sample at its place in time. An old sample beyond the limit falls out."""
        index = bisect.bisect_right(self._times, sample.t_utc_ns)
        self._times.insert(index, sample.t_utc_ns)
        self._samples.insert(index, sample)
        if len(self._samples) > self._max:
            del self._samples[0]
            del self._times[0]

    def add_record(self, record: SkyQualityRecord) -> bool:
        """Add the sample of a `sky_quality` record. Returns whether the record had one."""
        sample = sample_from_record(record)
        if sample is None:
            return False
        self.add(sample)
        return True

    def zero_points(self, since_utc_ns: int, until_utc_ns: int) -> tuple[ZeroPointSample, ...]:
        low = bisect.bisect_left(self._times, since_utc_ns)
        high = bisect.bisect_left(self._times, until_utc_ns)
        return tuple(self._samples[low:high])


@dataclass(frozen=True, slots=True)
class TransparencyOptions:
    """The rules of the reference and of the cloud flag."""

    window_days: float = 365.0
    quantile: float = 0.95
    min_samples: int = 20
    min_stars: int = 12  # a zero point that rests on fewer stars is not a reference
    max_rms_mag: float = 0.15  # nor is one with a larger scatter
    max_cloud_fraction: float = 0.2  # nor one from a cloudy frame
    cloud_flag_fraction: float = 0.3
    transparency_flag: float = 0.6
    night_split_utc_hour: float = 12.0
    # A frame without a zero point may use the median of this many hours while no reference
    # exists (`provisional_zero_point`). 0 turns the fallback off.
    fallback_hours: float = 6.0
    # A zero point (mag) that the owner trusts as the clear-sky reference. It replaces the history.
    pinned_zero_point: float | None = None

    def __post_init__(self) -> None:
        if (
            not 0.0 < self.quantile <= 1.0
            or self.window_days <= 0
            or self.min_samples < 1
            or not 0.0 <= self.fallback_hours < math.inf
            or (self.pinned_zero_point is not None and not math.isfinite(self.pinned_zero_point))
        ):
            raise ValueError("invalid transparency options")


@dataclass(frozen=True, slots=True)
class ZeroPointReference:
    """The zero point of the clearest conditions, and what it rests on.

    A `provisional` reference is the stand-in of `provisional_zero_point`: a median of the last
    few hours, which calibrates the sky of a frame without a zero point and sets no transparency.
    """

    zero_point_mag: float
    n_samples: int
    n_nights: int
    window_days: float
    quantile: float
    provisional: bool = False
    history_samples: int = 0  # the usable samples of the whole window, for "11 of 20" messages


def usable_samples(
    samples: Iterable[ZeroPointSample], options: TransparencyOptions
) -> list[ZeroPointSample]:
    """The samples that may set a reference: enough stars, a small scatter, and no clouds."""
    usable = []
    for sample in samples:
        if sample.n_stars < options.min_stars:
            continue
        if sample.rms_mag is not None and sample.rms_mag > options.max_rms_mag:
            continue
        if sample.cloud_fraction is not None and sample.cloud_fraction > options.max_cloud_fraction:
            continue
        if not math.isfinite(sample.zero_point_mag):
            continue
        usable.append(sample)
    return usable


def reference_zero_point(
    history: ZeroPointHistory,
    now_ns: int,
    options: TransparencyOptions | None = None,
) -> ZeroPointReference | None:
    """The reference zero point at a time, or `None` while the history is too short.

    The window ends just before `now_ns` and reaches `window_days` back. A pinned zero point
    (`pinned_zero_point`) is the reference at once, whatever the history holds.
    """
    cfg = options or TransparencyOptions()
    if cfg.pinned_zero_point is not None:
        return ZeroPointReference(
            zero_point_mag=float(cfg.pinned_zero_point),
            n_samples=0,
            n_nights=0,
            window_days=0.0,
            quantile=1.0,
        )
    since = now_ns - round(cfg.window_days * SECONDS_PER_DAY * NS_PER_S)
    usable = usable_samples(history.zero_points(since, now_ns), cfg)
    if len(usable) < cfg.min_samples:
        return None
    values = np.array([sample.zero_point_mag for sample in usable])
    nights = {night_label(sample.t_utc_ns, cfg.night_split_utc_hour) for sample in usable}
    return ZeroPointReference(
        zero_point_mag=float(np.quantile(values, cfg.quantile)),
        n_samples=len(usable),
        n_nights=len(nights),
        window_days=cfg.window_days,
        quantile=cfg.quantile,
    )


def provisional_zero_point(
    history: ZeroPointHistory,
    now_ns: int,
    options: TransparencyOptions | None = None,
) -> ZeroPointReference | None:
    """A stand-in for the reference zero point, from the last few hours, or `None`.

    The stand-in is the median of the usable zero points (`usable_samples`) in the window that
    reaches `fallback_hours` back and ends just before `now_ns`. One usable sample is enough.
    The result has `provisional` set, a `quantile` of 0.5, and a `window_days` of
    `fallback_hours / 24`. It serves a frame that has no zero point of its own, such as one
    that follows a lost pointing solution or a restart, so that its sky brightness does not drop
    out. It never sets a transparency, because a median of recent frames does not describe the
    clearest conditions. Use it only while `reference_zero_point` gives `None`.

    The result is `None` when `fallback_hours` is 0 (the fallback is off) or when the window
    holds no usable sample.
    """
    cfg = options or TransparencyOptions()
    if cfg.fallback_hours <= 0:
        return None
    since = now_ns - round(cfg.fallback_hours * SECONDS_PER_HOUR * NS_PER_S)
    usable = usable_samples(history.zero_points(since, now_ns), cfg)
    if not usable:
        return None
    values = np.array([sample.zero_point_mag for sample in usable])
    nights = {night_label(sample.t_utc_ns, cfg.night_split_utc_hour) for sample in usable}
    whole = now_ns - round(cfg.window_days * SECONDS_PER_DAY * NS_PER_S)
    return ZeroPointReference(
        zero_point_mag=float(np.median(values)),
        n_samples=len(usable),
        n_nights=len(nights),
        window_days=cfg.fallback_hours / 24.0,
        quantile=0.5,
        provisional=True,
        history_samples=len(usable_samples(history.zero_points(whole, now_ns), cfg)),
    )


def transparency(zero_point_mag: float, reference: ZeroPointReference) -> float:
    """`10 ** (-0.4 * (ZP_ref - ZP))`: 1 for the clearest conditions, less when clouds dim stars."""
    return float(10.0 ** (-0.4 * (reference.zero_point_mag - zero_point_mag)))


def cloud_flag(
    cloud_fraction: float | None, transparency_value: float | None, options: TransparencyOptions
) -> bool:
    """Whether a frame carries the `cloud` flag."""
    if cloud_fraction is not None and cloud_fraction >= options.cloud_flag_fraction:
        return True
    return transparency_value is not None and transparency_value < options.transparency_flag


# --- Detectability -------------------------------------------------------------------------


def min_detectable_flux_e(snr: float, noise_e_px: float, area_px2: float) -> float:
    """The flux in electrons that gives `snr` in an aperture of `area_px2` pixels.

    The aperture noise is the photon noise of the star plus `area_px2` pixels of background noise
    `noise_e_px` (read noise, dark, and sky). Solving `F / sqrt(F + A s^2) = snr` for `F` gives
    the flux that a detection needs.
    """
    s2 = snr**2
    return 0.5 * (s2 + math.sqrt(s2**2 + 4.0 * s2 * area_px2 * noise_e_px**2))


@dataclass(frozen=True, slots=True)
class LimitingMagnitude:
    """The limiting magnitude of a frame and how it was found.

    `status` is `measured` (half of the stars of a magnitude bin were detected at `value`),
    `predicted` (every bin held more than half, so `value` comes from the noise), or one of the
    reasons for no value: `too_few_stars`, `no_detections`.
    """

    value: float | None
    status: str
    n_stars: int


def _decreasing(fraction: FloatArray, weight: FloatArray) -> FloatArray:
    """The closest non-increasing sequence (pool adjacent violators, weighted)."""
    values = list(map(float, fraction))
    weights = list(map(float, weight))
    sizes = [1] * len(values)
    i = 0
    while i < len(values) - 1:
        if values[i] < values[i + 1]:  # a rise: pool the two blocks
            total = weights[i] + weights[i + 1]
            values[i] = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total
            weights[i] = total
            sizes[i] += sizes[i + 1]
            del values[i + 1], weights[i + 1], sizes[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out: list[float] = []
    for value, size in zip(values, sizes, strict=True):
        out.extend([value] * size)
    return np.array(out)


def limiting_magnitude(
    g_mag: FloatArray,
    found: BoolArray,
    *,
    bin_width_mag: float = 0.5,
    min_per_bin: int = 6,
    min_stars: int = 30,
    predicted_mag: float | None = None,
) -> LimitingMagnitude:
    """The magnitude where half of the catalog stars of the field are detected.

    `g_mag` holds the catalog magnitudes of the stars in the frame, and `found` whether each was
    detected. The stars go into bins `bin_width_mag` wide, and the bins with at least
    `min_per_bin` stars count. When every counted bin holds more than half detected stars, the
    limit lies beyond the catalog, and `predicted_mag` (from the noise) stands in for it when
    you pass one.
    """
    g = np.asarray(g_mag, dtype=np.float64)
    hit = np.asarray(found, dtype=np.bool_)
    ok = np.isfinite(g)
    g, hit = g[ok], hit[ok]
    if g.size < min_stars:
        return LimitingMagnitude(None, "too_few_stars", int(g.size))
    edges = np.arange(math.floor(g.min()), math.ceil(g.max()) + bin_width_mag, bin_width_mag)
    index = np.digitize(g, edges) - 1
    centers, fractions, counts = [], [], []
    for bin_index in range(len(edges) - 1):
        members = index == bin_index
        count = int(members.sum())
        if count >= min_per_bin:
            centers.append(float(g[members].mean()))
            fractions.append(float(hit[members].mean()))
            counts.append(count)
    if len(centers) < 2:
        return LimitingMagnitude(None, "too_few_stars", int(g.size))
    smooth = _decreasing(np.array(fractions), np.array(counts, dtype=np.float64))
    if smooth[0] < 0.5:
        return LimitingMagnitude(None, "no_detections", int(g.size))
    below = np.flatnonzero(smooth < 0.5)
    if below.size == 0:
        if predicted_mag is None:
            return LimitingMagnitude(None, "beyond_catalog", int(g.size))
        return LimitingMagnitude(predicted_mag, "predicted", int(g.size))
    j = int(below[0])
    f_high, f_low = float(smooth[j - 1]), float(smooth[j])
    c_high, c_low = centers[j - 1], centers[j]
    value = c_high + (f_high - 0.5) / (f_high - f_low) * (c_low - c_high)
    return LimitingMagnitude(value, "measured", int(g.size))
