"""The decisions about daylight, twilight, and clouds, as pure functions with hysteresis.

**Daylight.** Two inputs gate the `auto` state. The Sun's elevation from the ephemeris gates the
attempt: above `sun_elevation_limit_deg`, the scheduler does not try. The measured sky overrides
it: a background above `saturation_limit` of the saturation level at the shortest exposure forces
`safe`, whatever the ephemeris says. Each input has a stricter threshold for resuming than for
stopping, so a value that hovers at the limit does not flip the state every minute.

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
from seeingmon.profile import Profile
from seeingmon.scheduler.config import CloudConfig, DaylightConfig

# A frame with this many pixels or fewer is measured whole. A larger one is measured on a stride.
_MAX_SAMPLE_PIXELS = 65_536
_RAW8_FULL_SCALE = 255.0

REASON_DAYLIGHT = "daylight"
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
    data = frame.data
    stride = max(1, math.isqrt(data.size // _MAX_SAMPLE_PIXELS))
    median = float(np.median(data[::stride, ::stride]))
    return median / saturation_level_dn(frame, profile)


@dataclass(frozen=True, slots=True)
class DaylightDecision:
    """The verdict of the gate. `reason` is set when `allowed` is false."""

    allowed: bool
    reason: str | None
    twilight: bool


class DaylightGate:
    """Decide whether the scheduler may run `auto`, from the Sun and the measured sky."""

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
        sun_elevation_deg: float | None,
        background_fraction: float | None,
        running: bool,
    ) -> DaylightDecision:
        """Apply both inputs.

        `running` says whether the scheduler is in `auto` now. A running scheduler stops at the
        limits and a stopped one resumes only below the stricter resume values. A missing Sun
        elevation (no site) leaves the decision to the measurement. A missing measurement keeps a
        running scheduler running, and it stops a stopped one from starting.
        """
        config = self._config
        twilight = self.is_twilight(sun_elevation_deg)
        if sun_elevation_deg is not None:
            limit = config.sun_elevation_limit_deg
            if not running:
                limit -= config.sun_resume_margin_deg
            if sun_elevation_deg > limit:
                return DaylightDecision(False, REASON_DAYLIGHT, twilight)
        if background_fraction is None:
            if running:
                return DaylightDecision(True, None, twilight)
            return DaylightDecision(False, REASON_NO_MEASUREMENT, twilight)
        threshold = config.saturation_limit if running else config.resume_saturation
        if background_fraction >= threshold:
            return DaylightDecision(False, REASON_BRIGHT_SKY, twilight)
        return DaylightDecision(True, None, twilight)


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
