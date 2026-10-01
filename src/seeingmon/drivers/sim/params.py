"""Optics and sensor numbers of one readout mode.

`SimParams` holds everything the simulator needs to know about the telescope and one
readout mode of the camera: the pixel size, the aperture, the timing, and the gain table.
The defaults describe the reference hardware (an uncooled ZWO ASI294MM behind a 50 mm, f/5
guide scope), with the numbers in `docs/research-notes.md`. Build another camera by
replacing fields, or read the numbers from a profile with `SimParams.from_profile`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Final

from seeingmon.frames import Roi

ARCSEC_PER_RAD: Final = 206_264.80624709636

# The aperture that `mag0_electron_rate_e_per_s` refers to. Other apertures scale by area.
REFERENCE_APERTURE_M: Final = 0.050


@dataclass(frozen=True, slots=True)
class GainPoint:
    """One row of a gain table: what the sensor does at one gain setting."""

    gain: int
    e_per_adu: float
    read_noise_e: float
    full_well_e: float


# Read from ZWO's charts for the ASI294MM Pro (see "Gain, full well, and read noise" in the
# research notes). The camera reports the full well in electrons, and the ADC limits the
# usable signal at high gain, so the table lists the effective saturation level.
BIN1_GAIN_TABLE: Final = (
    GainPoint(0, 3.5, 2.65, 14_417.0),
    GainPoint(108, 1.0, 1.8, 4_200.0),
    GainPoint(270, 0.17, 1.38, 630.0),
)
BIN2_GAIN_TABLE: Final = (
    GainPoint(0, 4.05, 8.0, 66_387.0),
    GainPoint(119, 1.03, 6.2, 16_500.0),
    GainPoint(120, 0.88, 1.85, 15_000.0),
    GainPoint(300, 0.11, 1.3, 1_700.0),
)


@dataclass(frozen=True, slots=True)
class HighSpeedMode:
    """The ADC depth and timing of the camera's high-speed mode."""

    adc_bits: int
    row_time_s: float
    frame_overhead_s: float


@dataclass(frozen=True, slots=True)
class SensorAtGain:
    """The sensor numbers at one gain setting."""

    e_per_adu: float
    read_noise_e: float
    full_well_e: float


@dataclass(frozen=True, slots=True)
class SimParams:
    """The telescope and one readout mode of the camera.

    A pixel of this mode covers `binning` x `binning` native pixels. The sensor sums them, so
    dark current per pixel scales with `binning` squared. `width` and `height` are the full
    frame in pixels of this mode. `mag0_electron_rate_e_per_s` is the photoelectron rate of
    a magnitude-0 star through the reference 50 mm aperture, and the simulator scales it by
    the aperture area.
    """

    mode: str = "bin1"
    width: int = 8288
    height: int = 5644
    pixel_size_um: float = 2.315
    binning: int = 1
    row_time_s: float = 37.6e-6
    frame_overhead_s: float = 6.5e-3
    adc_bits: int = 12
    high_speed: HighSpeedMode | None = HighSpeedMode(10, 30.1e-6, 5.0e-3)
    gain_table: tuple[GainPoint, ...] = BIN1_GAIN_TABLE
    default_offset: int = 30
    focal_length_m: float = 0.250
    aperture_m: float = 0.050
    wavelength_m: float = 0.60e-6
    mag0_electron_rate_e_per_s: float = 4.6e7
    dark_current_e_per_s: float = 0.2
    dark_reference_c: float = 20.0
    dark_doubling_c: float = 6.0
    roi_width_multiple: int = 8
    roi_height_multiple: int = 2

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.binning < 1:
            raise ValueError("the frame size and the binning must be positive")
        if self.pixel_size_um <= 0 or self.focal_length_m <= 0 or self.aperture_m <= 0:
            raise ValueError("pixel size, focal length, and aperture must be positive")
        if self.wavelength_m <= 0 or self.row_time_s < 0 or self.frame_overhead_s < 0:
            raise ValueError("wavelength and timing must not be negative")
        if not self.gain_table:
            raise ValueError("gain_table must not be empty")
        gains = [row.gain for row in self.gain_table]
        if gains != sorted(gains):
            raise ValueError("gain_table must be sorted by gain")
        if not 8 <= self.adc_bits <= 16:
            raise ValueError("adc_bits must be between 8 and 16")

    # --- geometry ---

    @property
    def plate_scale_arcsec_per_px(self) -> float:
        """Arcseconds per pixel of this mode."""
        return ARCSEC_PER_RAD * self.pixel_size_um * 1e-6 / self.focal_length_m

    @property
    def pixel_rad(self) -> float:
        """Radians per pixel of this mode."""
        return self.pixel_size_um * 1e-6 / self.focal_length_m

    @property
    def full_roi(self) -> Roi:
        return Roi(0, 0, self.width, self.height)

    def normalize_roi(self, request: Roi | None) -> Roi:
        """Apply the vendor ROI rules to a request, as the SDK does.

        The width rounds down to a multiple of `roi_width_multiple`, the height to a
        multiple of `roi_height_multiple`, and the ROI shifts to stay inside the frame.
        Raises `ValueError` for a ROI that cannot fit.
        """
        if request is None:
            return self.full_roi
        width = max(
            self.roi_width_multiple,
            request.width // self.roi_width_multiple * self.roi_width_multiple,
        )
        height = max(
            self.roi_height_multiple,
            request.height // self.roi_height_multiple * self.roi_height_multiple,
        )
        if width > self.width or height > self.height:
            raise ValueError("the ROI is larger than the frame")
        return Roi(
            min(request.x, self.width - width),
            min(request.y, self.height - height),
            width,
            height,
        )

    # --- timing ---

    def effective(self, *, high_speed: bool) -> SimParams:
        """The parameters with the high-speed mode applied or removed."""
        if not high_speed:
            return self
        if self.high_speed is None:
            raise ValueError(f"mode {self.mode!r} has no high-speed variant")
        speed = self.high_speed
        return replace(
            self,
            adc_bits=speed.adc_bits,
            row_time_s=speed.row_time_s,
            frame_overhead_s=speed.frame_overhead_s,
            high_speed=None,
        )

    def readout_time_s(self, rows: int) -> float:
        """Time to read `rows` rows: the frame overhead plus the row time for each row."""
        return self.frame_overhead_s + rows * self.row_time_s

    # --- sensor ---

    def sensor_at(self, gain: int) -> SensorAtGain:
        """The electrons per ADU, read noise, and full well at a gain setting.

        Between table rows, the values interpolate linearly in the logarithm of the value
        against the gain. A gain outside the table takes the nearest row.
        """
        rows = self.gain_table
        if gain <= rows[0].gain:
            row = rows[0]
            return SensorAtGain(row.e_per_adu, row.read_noise_e, row.full_well_e)
        if gain >= rows[-1].gain:
            row = rows[-1]
            return SensorAtGain(row.e_per_adu, row.read_noise_e, row.full_well_e)
        for low, high in pairwise(rows):
            if low.gain <= gain <= high.gain:
                span = high.gain - low.gain
                t = (gain - low.gain) / span if span else 1.0
                return SensorAtGain(
                    _log_lerp(low.e_per_adu, high.e_per_adu, t),
                    _log_lerp(low.read_noise_e, high.read_noise_e, t),
                    _log_lerp(low.full_well_e, high.full_well_e, t),
                )
        raise AssertionError("unreachable: the gain lies inside the table")

    def dark_rate_e_per_s(self, temperature_c: float) -> float:
        """Dark current per pixel of this mode at a sensor temperature."""
        per_native = self.dark_current_e_per_s * math.pow(
            2.0, (temperature_c - self.dark_reference_c) / self.dark_doubling_c
        )
        return per_native * self.binning**2

    def mag0_rate_e_per_s(self) -> float:
        """Photoelectrons per second from a magnitude-0 star, scaled to this aperture."""
        return self.mag0_electron_rate_e_per_s * (self.aperture_m / REFERENCE_APERTURE_M) ** 2

    def sky_rate_e_per_s_px(self, sky_mag_arcsec2: float) -> float:
        """Photoelectrons per second per pixel from a sky of the given surface brightness."""
        per_arcsec2 = self.mag0_rate_e_per_s() * math.pow(10.0, -0.4 * sky_mag_arcsec2)
        return per_arcsec2 * self.plate_scale_arcsec_per_px**2

    # --- construction ---

    @classmethod
    def reference(cls, mode: str = "bin1") -> SimParams:
        """The reference hardware in one readout mode: `bin1` or `bin2`."""
        if mode == "bin1":
            return cls()
        if mode == "bin2":
            return cls(
                mode="bin2",
                width=4144,
                height=2822,
                pixel_size_um=4.63,
                binning=2,
                row_time_s=21.3e-6,
                frame_overhead_s=1.4e-3,
                adc_bits=14,
                high_speed=HighSpeedMode(12, 18.2e-6, 1.2e-3),
                gain_table=BIN2_GAIN_TABLE,
            )
        raise ValueError(f"unknown reference mode {mode!r}: use 'bin1' or 'bin2'")


def reference_modes() -> dict[str, SimParams]:
    """Both readout modes of the reference hardware, by name."""
    return {name: SimParams.reference(name) for name in ("bin1", "bin2")}


def _log_lerp(low: float, high: float, t: float) -> float:
    return math.exp(math.log(low) + (math.log(high) - math.log(low)) * t)
