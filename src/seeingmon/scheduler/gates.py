"""The decisions about daylight, twilight, and clouds, as pure functions with hysteresis.

**Daylight.** The measured sky alone gates the `auto` state, and the Sun's elevation gates nothing:
the celestial pole is always at least 66.5 degrees from the Sun, so the Sun never enters the
field. The gate reads a brightness frame (the watch frame in `safe`, the short survey frame in
`auto`), and `read_sky` derives through the profile the background that the fast stream would
have at the profile's shortest exposure. A background above `saturation_limit` of saturation there
forces `safe`, because nothing can be measured even at that exposure. A brightness frame that has
clipped (its median reached `brightness_clip_fraction` of its own saturation level) shows only that
the sky is at least that bright, so it counts as too bright. With a 1 ms bin2 brightness frame,
the frame clips while the fast stream would still see less than 4% of saturation, so in practice
the clip stops `auto`. Both rules have a stricter threshold for resuming than for stopping
(`resume_saturation` for the fast background, `brightness_resume_fraction` for the brightness
frame), so a sky that hovers at a limit does not flip the state every minute.

**Twilight.** While the Sun is above `twilight_elevation_deg`, windows and survey results carry
the `twilight` flag. At some latitudes the Sun stays above that elevation for weeks.

**Clouds.** The survey analysis reports a cloud fraction. At or above `threshold`, the cloud
response starts. At or below `clear_threshold`, it ends. A fraction between the two keeps the
current state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
    data = frame.data
    stride = max(1, math.isqrt(data.size // _MAX_SAMPLE_PIXELS))
    return float(np.median(data[::stride, ::stride]))


def _e_per_dn(frame: Frame, profile: Profile) -> float:
    """The electrons of one count of a frame, in the container that the frame uses.

    A 16-bit container holds the ADC value of the readout mode in its high bits, and an 8-bit one
    its top 8 bits, as in the fast path.
    """
    readout = profile.mode(frame.mode)
    container_bits = 8 if frame.pixel_format is PixelFormat.RAW8 else 16
    return derived.e_per_adu(readout, frame.gain) * 2.0 ** (readout.adc_bits - container_bits)


@dataclass(frozen=True, slots=True)
class SkyReading:
    """What one brightness frame says about the sky, for the daylight gate.

    `frame_fraction` is the median of the brightness frame as a share of its own saturation level.
    `fast_fraction` is the background that the fast stream would have at its shortest exposure:
    the sky electrons of a fast-mode pixel over the full well at the fast gain. `clipped` says
    that the brightness frame reached the clip level, so the sky is at least that bright, and
    `fast_fraction` is only a lower bound.
    """

    frame_fraction: float
    fast_fraction: float
    clipped: bool

    @property
    def gate_fraction(self) -> float:
        """The share of saturation that the gate compares: 1 for a clipped brightness frame."""
        return 1.0 if self.clipped else self.fast_fraction


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
    )


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

    def evaluate(
        self,
        *,
        background_fraction: float | None,
        running: bool,
        frame_fraction: float | None = None,
    ) -> DaylightDecision:
        """Decide from the background of the fast stream at its shortest exposure.

        `background_fraction` is `SkyReading.gate_fraction`, and `running` says whether the
        scheduler is in `auto` now. A running scheduler stops at `saturation_limit` (a clipped
        brightness frame reaches it), and a stopped one resumes only below the stricter
        `resume_saturation`. `frame_fraction` is `SkyReading.frame_fraction`: a stopped scheduler
        also needs it below `brightness_resume_fraction`, the hysteresis of the clip. A missing
        measurement keeps a running scheduler running, and it stops a stopped one from starting.
        """
        config = self._config
        if background_fraction is None:
            if running:
                return DaylightDecision(True, None)
            return DaylightDecision(False, REASON_NO_MEASUREMENT)
        if (
            not running
            and frame_fraction is not None
            and frame_fraction >= config.brightness_resume_fraction
        ):
            return DaylightDecision(False, REASON_BRIGHT_SKY)
        threshold = config.saturation_limit if running else config.resume_saturation
        if background_fraction >= threshold:
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
