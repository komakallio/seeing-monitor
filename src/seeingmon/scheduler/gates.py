"""The decisions about daylight, twilight, and clouds, as pure functions with hysteresis.

**Daylight.** The measured sky alone gates the `auto` state, and the Sun's elevation gates nothing:
the celestial pole is always at least 66.5 degrees from the Sun, so the Sun never enters the
field. The gate judges what the fast stream sees: its background at the profile's shortest
exposure, as a share of saturation (`FastSky`). Above `saturation_limit` nothing can be measured
even at that exposure, so the scheduler goes to `safe`, and it resumes `auto` only below the
stricter `resume_saturation`, so that a sky that hovers at the limit does not flip the state every
minute. Two sources give the background:

- **A brightness frame.** `read_sky` derives the background through the profile from the watch
  frame in `safe`, and from the 1 ms frame of each survey step in `auto`. Both take 1 ms in bin2,
  and they clip while the fast stream would still see less than 4% of saturation. A second watch
  frame at the profile's shortest exposure (`[scheduler.watch] bright_exposure_us`, 32 us) then
  measures the sky, because it does not clip in daylight.
- **The fast stream itself.** In `auto`, each search burst and each window reports its background,
  which scales to the shortest exposure in proportion.

A frame whose median reaches `brightness_clip_fraction` of its saturation level has clipped, so it
gives only a lower bound: the sky is at least that bright. The gate takes the largest estimate. In
`auto`, a lower bound under the limit decides nothing, and the scheduler then takes a watch frame
at the bright exposure before the long survey exposure. In `safe`, it keeps the scheduler there.

**Twilight.** While the Sun is above `twilight_elevation_deg`, windows and survey results carry
the `twilight` flag. At some latitudes the Sun stays above that elevation for weeks.

**Clouds.** The survey analysis reports a cloud fraction. At or above `threshold`, the cloud
response starts. At or below `clear_threshold`, it ends. A fraction between the two keeps the
current state.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.frames import Frame, PixelFormat
from seeingmon.profile import Profile, derived
from seeingmon.scheduler.config import CloudConfig, DaylightConfig

# A frame with this many pixels or fewer is measured whole. A larger one is measured on a stride.
_MAX_SAMPLE_PIXELS = 65_536
_RAW8_FULL_SCALE = 255.0

REASON_BRIGHT_SKY = "bright_sky"
REASON_NO_MEASUREMENT = "no_measurement"


def saturation_level_dn(frame: Frame, profile: Profile) -> float:
    """The saturation level of a frame in the counts that the frame carries.

    A 16-bit frame holds the ADC value in the high bits, so the level comes from the profile for
    the frame's readout mode and gain. The vendor does not document the 8-bit scaling, so an
    8-bit frame uses its full scale.
    """
    if frame.pixel_format is PixelFormat.RAW8:
        return _RAW8_FULL_SCALE
    return profile.saturation(frame.mode, frame.gain).container_dn


def sky_background_fraction(frame: Frame, profile: Profile) -> float:
    """The median of a frame as a share of the saturation level for its mode and gain.

    A large frame is measured on a regular stride of at most about 65,000 pixels, because the
    gate needs a robust level and not every pixel.
    """
    return median_dn(frame) / saturation_level_dn(frame, profile)


def median_dn(frame: Frame) -> float:
    """The median of a frame in its own counts, on a stride of at most about 65,000 pixels."""
    return median_of(frame.data)


def median_of(data: npt.NDArray[Any]) -> float:
    """The median of an image, on a regular stride of at most about 65,000 pixels."""
    stride = max(1, math.isqrt(data.size // _MAX_SAMPLE_PIXELS))
    return float(np.median(data[::stride, ::stride]))


def e_per_dn(profile: Profile, mode: str, gain: int, pixel_format: PixelFormat) -> float:
    """The electrons of one count of a readout mode and gain, in the container of a pixel format.

    A 16-bit container holds the ADC value of the readout mode in its high bits, and an 8-bit one
    its top 8 bits, as in the fast path.
    """
    readout = profile.mode(mode)
    container_bits = 8 if pixel_format is PixelFormat.RAW8 else 16
    return derived.e_per_adu(readout, gain) * 2.0 ** (readout.adc_bits - container_bits)


def _e_per_dn(frame: Frame, profile: Profile) -> float:
    """The electrons of one count of a frame, in the container that the frame uses."""
    return e_per_dn(profile, frame.mode, frame.gain, frame.pixel_format)


@dataclass(frozen=True, slots=True)
class SkyReading:
    """What one brightness frame says about the sky, for the daylight gate.

    `frame_fraction` is the median of the brightness frame as a share of its own saturation level.
    `fast_fraction` is the background that the fast stream would have at its shortest exposure:
    the sky electrons of a fast-mode pixel over the full well at the fast gain. `clipped` says
    that the brightness frame reached the clip level (`brightness_clip_fraction`), so the sky is
    at least that bright, and `fast_fraction` is only a lower bound. `median_dn` is the median of
    the frame in its own counts.
    """

    frame_fraction: float
    fast_fraction: float
    clipped: bool
    median_dn: float = 0.0


def read_sky(
    frame: Frame,
    profile: Profile,
    *,
    fast_mode: str,
    fast_exposure_us: float,
    fast_gain: int,
    clip_fraction: float,
    offset_dn: float = 0.0,
) -> SkyReading:
    """Derive the background of the fast stream from a brightness frame, through the profile.

    The median of the brightness frame above its offset (`offset_dn`, the black level in the
    frame's own counts) gives the sky electrons of one of its pixels, through the conversion gain
    of its readout mode and gain. Per pixel area and per microsecond of its exposure, that is the
    sky rate on the sensor, and times the area of a fast-mode pixel and `fast_exposure_us` it is
    the sky of one fast pixel. Its share of the full well of the fast mode at `fast_gain` is
    `fast_fraction`.

    The profile has no offset, because the camera reports it at run time, and the default of 0
    counts it as sky. That errs toward a brighter sky by the offset scaled to the fast exposure:
    with the simulator's offset, 0.03% of saturation for the reference profile, a 1 ms brightness
    frame, and 32 us.
    """
    median = median_dn(frame)
    frame_fraction = median / saturation_level_dn(frame, profile)
    brightness = profile.mode(frame.mode)
    fast = profile.mode(fast_mode)
    sky_e = max(median - offset_dn, 0.0) * _e_per_dn(frame, profile)
    area = (fast.pixel_size_um / brightness.pixel_size_um) ** 2
    fast_e = sky_e * area * fast_exposure_us / frame.exposure_us
    full_well = derived.saturation(fast, fast_gain).full_well_e
    return SkyReading(
        frame_fraction=frame_fraction,
        fast_fraction=fast_e / full_well,
        clipped=frame_fraction >= clip_fraction,
        median_dn=median,
    )


@dataclass(frozen=True, slots=True)
class FastSky:
    """The background of the fast stream at the profile's shortest exposure, for the gate.

    `fraction` is a share of saturation. `bound` says that it comes from a frame that clipped, so
    the sky is at least that bright.
    """

    fraction: float
    bound: bool = False

    @classmethod
    def from_reading(cls, reading: SkyReading) -> FastSky:
        """The estimate of a brightness frame: a lower bound when the frame clipped."""
        return cls(reading.fast_fraction, bound=reading.clipped)

    @classmethod
    def from_fast(
        cls, fraction: float, exposure_us: float, shortest_us: float, *, clipped: bool
    ) -> FastSky:
        """The estimate of a burst or a window of the fast stream, scaled to the shortest exposure.

        The background grows in proportion to the exposure. `fraction` counts the camera's offset
        as sky, as everywhere in the loop, and the offset scales down here with the exposure, where
        a frame at the shortest exposure would hold all of it: 0.7% of saturation with the
        simulator's offset, far below the limits of the gate.
        """
        return cls(fraction * shortest_us / exposure_us, bound=clipped)


def combine(estimates: Iterable[FastSky | None]) -> FastSky | None:
    """The estimate that the gate judges: the largest, and a bound only when it is one.

    A bound larger than every measurement says that the sky is at least that bright, so the result
    is that bound. A measurement at or above every bound stands as a measurement.
    """
    known = [estimate for estimate in estimates if estimate is not None]
    if not known:
        return None
    measured = [e.fraction for e in known if not e.bound]
    bounds = [e.fraction for e in known if e.bound]
    if measured and (not bounds or max(measured) >= max(bounds)):
        return FastSky(max(measured))
    return FastSky(max(bounds), bound=True)


@dataclass(frozen=True, slots=True)
class DaylightDecision:
    """The verdict of the gate. `reason` is set when `allowed` is false."""

    allowed: bool
    reason: str | None


class DaylightGate:
    """Decide whether the scheduler may run `auto`, from the measured sky."""

    def __init__(self, config: DaylightConfig) -> None:
        self._config = config

    def is_twilight(self, sun_elevation_deg: float | None) -> bool:
        """Whether the Sun is above the twilight limit. Without a site, no flag applies."""
        return (
            sun_elevation_deg is not None
            and sun_elevation_deg > self._config.twilight_elevation_deg
        )

    def threshold(self, *, running: bool) -> float:
        """The share of saturation that stops a running scheduler, or below which a stopped one
        resumes."""
        config = self._config
        return config.saturation_limit if running else config.resume_saturation

    def evaluate(self, sky: FastSky | None, *, running: bool) -> DaylightDecision:
        """Decide from the background of the fast stream at its shortest exposure.

        `sky` is the estimate that `combine` gives, and `running` says whether the scheduler is in
        `auto` now. A running scheduler stops at `saturation_limit`, and a stopped one resumes only
        below the stricter `resume_saturation`. A missing estimate, or a lower bound under the
        threshold, keeps a running scheduler running, and it keeps a stopped one stopped.
        """
        threshold = self.threshold(running=running)
        if sky is None:
            if running:
                return DaylightDecision(True, None)
            return DaylightDecision(False, REASON_NO_MEASUREMENT)
        if sky.fraction >= threshold:
            return DaylightDecision(False, REASON_BRIGHT_SKY)
        if sky.bound and not running:
            return DaylightDecision(False, REASON_BRIGHT_SKY)
        return DaylightDecision(True, None)


class CloudTracker:
    """Follow the cloud fraction with hysteresis, and say what the response changes."""

    def __init__(self, config: CloudConfig) -> None:
        self._config = config
        self._active = False
        self._fraction: float | None = None

    @property
    def active(self) -> bool:
        """Whether the cloud response applies now."""
        return self._active

    @property
    def fraction(self) -> float | None:
        """The latest cloud fraction, or `None` before the first result."""
        return self._fraction

    def update(self, fraction: float | None) -> bool:
        """Take a result. Returns `True` when the cloud response just started or ended.

        A result of `None` (the analysis cannot tell) changes nothing.
        """
        if fraction is None:
            return False
        self._fraction = fraction
        was_active = self._active
        if fraction >= self._config.threshold:
            self._active = True
        elif fraction <= self._config.clear_threshold:
            self._active = False
        return self._active != was_active

    def fast_window_s(self, normal_s: float) -> float:
        """The length of the fast period: the cloud value while the response applies."""
        return self._config.fast_window_s if self._active else normal_s

    def survey_cadence_s(self, normal_s: float) -> float:
        """The survey cadence: the cloud value while the response applies."""
        return self._config.survey_cadence_s if self._active else normal_s
