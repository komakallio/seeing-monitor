"""Derived values of a profile, as pure functions of its models.

Each function takes the readout mode (and the optics or the limits when it needs them) and
returns a number or a small immutable result. The functions read no files and keep no state.
`Profile` offers each one as a method, so `profile.plate_scale_arcsec_per_px("bin1")` and
`plate_scale_arcsec_per_px(mode, optics)` give the same answer.

Always name the readout mode when you quote a plate scale or a pixel size, because a bin1
pixel and a bin2 pixel differ by a factor of two.

**Gain tables.** The mode's gain-0 values and its `gain_points` form one table. Between two
points, electrons per ADU interpolates log-linearly in gain and the read noise interpolates
linearly in gain. A point with `step` set starts a new segment, so values never interpolate
across it, and a gain below the step takes the values of the point before it. Above the last
point, the values stay at the last point, because the table ends where the vendor's chart ends.

**Saturation.** A pixel saturates at the smaller of two limits: the full well and the ADC
full scale. The full well in electrons is the smaller of the gain-0 full well and electrons
per ADU times the ADC full scale. The saturation level in native ADC counts follows from it.
The vendor SDK places the ADC value in the high bits of a 16-bit container, so the level in
container counts is the native level times 2^(16 - ADC bits). The levels ignore the black-level
offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from seeingmon.frames import PixelFormat, Roi
from seeingmon.profile.errors import ProfileError

if TYPE_CHECKING:
    from seeingmon.profile.models import Limits, Optics, Photometry, ReadoutMode

ARCSEC_PER_RAD = 206264.80624709636
AIRY_FWHM_FACTOR = 1.029  # the FWHM of an Airy pattern is 1.029 lambda / D
CONTAINER_BITS = 16  # RAW16 carries the ADC value in the high bits of 16 bits
_MM_PER_UM = 1e-3
_M_PER_NM = 1e-9
_M_PER_MM = 1e-3
_S_PER_US = 1e-6
_S_PER_MS = 1e-3
_ARCSEC_PER_ARCMIN = 60.0
_NS_PER_US = 1000.0


@dataclass(frozen=True, slots=True)
class FieldOfView:
    """The angular size of the sensor in one readout mode, in degrees."""

    width_deg: float
    height_deg: float
    diagonal_deg: float


@dataclass(frozen=True, slots=True)
class Saturation:
    """Where a pixel saturates at one gain.

    `full_well_e` is the effective full well in electrons. `native_dn` counts in the ADC's own
    units, and `container_dn` counts in a 16-bit container, which holds the ADC value in the high
    bits. `limited_by` says which limit applies: the ADC full scale or the full well.
    """

    gain: float
    full_well_e: float
    native_dn: float
    container_dn: float
    limited_by: Literal["adc", "well"]


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, int):  # an integer is finite, and a huge one does not convert to a float
        return
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


# --- Geometry ----------------------------------------------------------------------------


def plate_scale_arcsec_per_px(mode: ReadoutMode, optics: Optics) -> float:
    """The plate scale of a readout mode: 206265 x pixel size / focal length, in arcsec/pixel."""
    return ARCSEC_PER_RAD * mode.pixel_size_um * _MM_PER_UM / optics.focal_length_mm


def sensor_size_mm(mode: ReadoutMode) -> tuple[float, float]:
    """The width and height of the sensor area that the readout mode covers, in millimetres."""
    return (
        mode.width_px * mode.pixel_size_um * _MM_PER_UM,
        mode.height_px * mode.pixel_size_um * _MM_PER_UM,
    )


def sensor_diagonal_mm(mode: ReadoutMode) -> float:
    """The diagonal of the sensor area that the readout mode covers, in millimetres."""
    width_mm, height_mm = sensor_size_mm(mode)
    return math.hypot(width_mm, height_mm)


def field_of_view(mode: ReadoutMode, optics: Optics) -> FieldOfView:
    """The field of view: 2 atan(extent / (2 focal length)) for the width, height, and diagonal."""
    width_mm, height_mm = sensor_size_mm(mode)

    def angle_deg(extent_mm: float) -> float:
        return math.degrees(2.0 * math.atan(extent_mm / (2.0 * optics.focal_length_mm)))

    return FieldOfView(
        width_deg=angle_deg(width_mm),
        height_deg=angle_deg(height_mm),
        diagonal_deg=angle_deg(math.hypot(width_mm, height_mm)),
    )


def usable_radius_px(mode: ReadoutMode, optics: Optics) -> float | None:
    """The radius of the usable image circle in pixels of the mode, or `None` for the full sensor.

    The circle is centered on the sensor.
    """
    if optics.image_circle_diameter_mm is None:
        return None
    return optics.image_circle_diameter_mm / 2.0 / (mode.pixel_size_um * _MM_PER_UM)


# --- Optics and sampling -----------------------------------------------------------------


def _wavelength_nm(optics: Optics, wavelength_nm: float | None) -> float:
    wavelength = optics.wavelength_nm if wavelength_nm is None else wavelength_nm
    _require_positive("wavelength_nm", wavelength)
    return wavelength


def airy_fwhm_arcsec(optics: Optics, wavelength_nm: float | None = None) -> float:
    """The FWHM of the diffraction-limited star image: 1.029 lambda / D, in arcsec.

    The wavelength defaults to the profile's effective wavelength.
    """
    wavelength = _wavelength_nm(optics, wavelength_nm)
    return (
        AIRY_FWHM_FACTOR
        * wavelength
        * _M_PER_NM
        / (optics.aperture_mm * _M_PER_MM)
        * ARCSEC_PER_RAD
    )


def airy_fwhm_px(mode: ReadoutMode, optics: Optics, wavelength_nm: float | None = None) -> float:
    """The Airy FWHM in pixels of the readout mode."""
    return airy_fwhm_arcsec(optics, wavelength_nm) / plate_scale_arcsec_per_px(mode, optics)


def sampling_ratio(mode: ReadoutMode, optics: Optics, wavelength_nm: float | None = None) -> float:
    """The pixel size over lambda times the f-number.

    A centroid shows no phase bias when the ratio is below 1. The wavelength defaults to the
    profile's effective wavelength.
    """
    wavelength_um = _wavelength_nm(optics, wavelength_nm) * 1e-3
    return mode.pixel_size_um / (wavelength_um * optics.f_number)


# --- ROI rules ---------------------------------------------------------------------------


def _nearest_multiple(value: float, multiple: int) -> int:
    """Round `value` to the nearest multiple of `multiple`. A tie rounds up."""
    if isinstance(value, int):
        return (value + multiple // 2) // multiple * multiple
    return math.floor(value / multiple + 0.5) * multiple


def _round_half_up(value: float) -> int:
    return value if isinstance(value, int) else math.floor(value + 0.5)


def _fit_size(requested: float, multiple: int, frame: int) -> int:
    """The nearest multiple of `multiple` to `requested`, between one step and the frame size."""
    largest = frame // multiple * multiple
    if largest < multiple:
        raise ValueError(f"a frame of {frame} pixels is smaller than the ROI step of {multiple}")
    return max(multiple, min(_nearest_multiple(requested, multiple), largest))


def roi_size_px(
    mode: ReadoutMode, optics: Optics, limits: Limits, full_width_arcmin: float
) -> tuple[int, int]:
    """The ROI size (width, height) in pixels for a square patch of sky of the given full width.

    The width rounds to the nearest multiple of `limits.roi_width_multiple` and the height to
    the nearest multiple of `limits.roi_height_multiple`. A tie rounds up. Both clamp to the
    frame, so the result is always a valid ROI size.
    """
    _require_positive("full_width_arcmin", full_width_arcmin)
    pixels = full_width_arcmin * _ARCSEC_PER_ARCMIN / plate_scale_arcsec_per_px(mode, optics)
    return (
        _fit_size(pixels, limits.roi_width_multiple, mode.width_px),
        _fit_size(pixels, limits.roi_height_multiple, mode.height_px),
    )


def clamp_roi(
    mode: ReadoutMode, limits: Limits, x: float, y: float, width: float, height: float
) -> Roi:
    """Round and clamp a requested ROI so that it follows the rules and fits in the frame.

    The width and height round to the nearest allowed multiple (a tie rounds up) and clamp
    between one step and the frame size. The origin rounds to the nearest pixel, then moves
    the least distance that keeps the ROI inside the frame. Any finite request is accepted,
    including a negative origin or a size larger than the frame.
    """
    for name, value in (("x", x), ("y", y), ("width", width), ("height", height)):
        _require_finite(name, value)
    fitted_width = _fit_size(width, limits.roi_width_multiple, mode.width_px)
    fitted_height = _fit_size(height, limits.roi_height_multiple, mode.height_px)
    return Roi(
        x=min(max(_round_half_up(x), 0), mode.width_px - fitted_width),
        y=min(max(_round_half_up(y), 0), mode.height_px - fitted_height),
        width=fitted_width,
        height=fitted_height,
    )


# --- Gain, noise, and saturation ---------------------------------------------------------


def _gain_values(mode: ReadoutMode, gain: float) -> tuple[float, float]:
    """The electrons per ADU and the read noise in electrons at `gain`."""
    _require_finite("gain", gain)
    if gain < 0:
        raise ValueError(f"gain must not be negative, got {gain}")
    rows = [
        (0, mode.e_per_adu_gain0, mode.read_noise_gain0_e, False),
        *((p.gain, p.e_per_adu, p.read_noise_e, p.step) for p in mode.gain_points),
    ]
    index = max(i for i, row in enumerate(rows) if row[0] <= gain)
    row_gain, e_per_adu, read_noise, _ = rows[index]
    if index == len(rows) - 1:  # at or above the last row
        return e_per_adu, read_noise
    next_gain, next_e_per_adu, next_read_noise, next_is_step = rows[index + 1]
    if next_is_step:  # the values jump at the next point, so this segment ends here
        return e_per_adu, read_noise
    fraction = (gain - row_gain) / (next_gain - row_gain)
    return (
        e_per_adu * math.exp(fraction * math.log(next_e_per_adu / e_per_adu)),
        read_noise + fraction * (next_read_noise - read_noise),
    )


def e_per_adu(mode: ReadoutMode, gain: float) -> float:
    """The conversion gain in electrons per ADU (in the ADC's native counts) at `gain`."""
    return _gain_values(mode, gain)[0]


def read_noise_e(mode: ReadoutMode, gain: float) -> float:
    """The read noise in electrons at `gain`."""
    return _gain_values(mode, gain)[1]


def adc_full_scale(mode: ReadoutMode) -> int:
    """The largest value that the ADC produces: 2^bits - 1, in native counts."""
    return (1 << mode.adc_bits) - 1


def full_well_e(mode: ReadoutMode, gain: float) -> float:
    """The effective full well in electrons at `gain`.

    It is the smaller of the gain-0 full well and electrons per ADU times the ADC full scale,
    because the ADC clips before the well fills at high gain.
    """
    return min(mode.full_well_gain0_e, e_per_adu(mode, gain) * adc_full_scale(mode))


def saturation(mode: ReadoutMode, gain: float) -> Saturation:
    """The saturation level at `gain`, in native ADC counts and in 16-bit container counts."""
    conversion = e_per_adu(mode, gain)
    full_scale = adc_full_scale(mode)
    well_dn = mode.full_well_gain0_e / conversion
    native_dn: float
    limited_by: Literal["adc", "well"]
    if well_dn < full_scale:
        native_dn, limited_by = well_dn, "well"
    else:
        native_dn, limited_by = float(full_scale), "adc"
    return Saturation(
        gain=gain,
        full_well_e=native_dn * conversion,
        native_dn=native_dn,
        container_dn=native_dn * 2.0 ** (CONTAINER_BITS - mode.adc_bits),
        limited_by=limited_by,
    )


# --- Timing and data rate ----------------------------------------------------------------


def row_time_ns(mode: ReadoutMode) -> int:
    """The row time in nanoseconds. A star in ROI row `r` exposes `r` row times later."""
    return round(mode.row_time_us * _NS_PER_US)


def _require_rows(mode: ReadoutMode, roi_height_px: int) -> None:
    if not 1 <= roi_height_px <= mode.height_px:
        raise ValueError(
            f"the ROI height must be 1 to {mode.height_px} pixels in mode {mode.name!r}, "
            f"got {roi_height_px}"
        )


def readout_time_s(mode: ReadoutMode, roi_height_px: int) -> float:
    """The time to read a frame: the frame overhead plus the ROI rows times the row time."""
    _require_rows(mode, roi_height_px)
    return mode.frame_overhead_ms * _S_PER_MS + roi_height_px * mode.row_time_us * _S_PER_US


def frame_period_s(mode: ReadoutMode, roi_height_px: int, exposure_us: float) -> float:
    """The time between frames: the larger of the exposure and the readout time."""
    _require_positive("exposure_us", exposure_us)
    return max(exposure_us * _S_PER_US, readout_time_s(mode, roi_height_px))


def max_frame_rate_hz(mode: ReadoutMode, roi_height_px: int, exposure_us: float) -> float:
    """The highest frame rate for an ROI height and an exposure, in frames per second."""
    return 1.0 / frame_period_s(mode, roi_height_px, exposure_us)


def frame_bytes(
    roi_width_px: int, roi_height_px: int, pixel_format: PixelFormat = PixelFormat.RAW16
) -> int:
    """The size of one frame in bytes, without a header."""
    if roi_width_px < 1 or roi_height_px < 1:
        raise ValueError(f"the ROI must be at least 1 x 1, got {roi_width_px} x {roi_height_px}")
    return roi_width_px * roi_height_px * (pixel_format.value // 8)


def data_rate_bytes_per_s(
    mode: ReadoutMode,
    roi_width_px: int,
    roi_height_px: int,
    exposure_us: float,
    pixel_format: PixelFormat = PixelFormat.RAW16,
) -> float:
    """The pixel data rate at the highest frame rate, in bytes per second."""
    if roi_width_px > mode.width_px:
        raise ValueError(
            f"the ROI width must be at most {mode.width_px} pixels in mode {mode.name!r}, "
            f"got {roi_width_px}"
        )
    size = frame_bytes(roi_width_px, roi_height_px, pixel_format)
    return size * max_frame_rate_hz(mode, roi_height_px, exposure_us)


# --- Photometry --------------------------------------------------------------------------


def star_electron_rate_e_per_s(photometry: Photometry | None, magnitude: float) -> float:
    """The photoelectron rate of a star of the given magnitude, in electrons per second.

    The rate scales from the profile's magnitude-0 rate, which is an estimate. Raises
    `ProfileError` when the profile has no such rate.
    """
    _require_finite("magnitude", magnitude)
    if photometry is None or photometry.mag0_electron_rate_e_per_s is None:
        raise ProfileError("the profile has no photometry.mag0_electron_rate_e_per_s")
    return photometry.mag0_electron_rate_e_per_s * math.pow(10.0, -0.4 * magnitude)


def dark_current_e_per_s_per_px(photometry: Photometry | None, temperature_c: float) -> float:
    """The prior for the dark current at a sensor temperature, in electrons per second per pixel.

    The prior interpolates the profile's table log-linearly in temperature, and it continues
    along the end segments outside the table. A fitted dark model replaces it once you have
    dark frames. Raises `ProfileError` when the profile has no table.
    """
    _require_finite("temperature_c", temperature_c)
    if photometry is None or not photometry.dark_current:
        raise ProfileError("the profile has no photometry.dark_current table")
    points = photometry.dark_current
    index = max(
        (i for i in range(len(points) - 1) if points[i].temperature_c <= temperature_c), default=0
    )
    low, high = points[index], points[index + 1]
    log_slope = math.log(high.e_per_s_per_px / low.e_per_s_per_px) / (
        high.temperature_c - low.temperature_c
    )
    try:
        return low.e_per_s_per_px * math.exp(log_slope * (temperature_c - low.temperature_c))
    except OverflowError:
        raise ProfileError(
            f"the dark-current table does not extrapolate to {temperature_c} C: "
            "check its end points"
        ) from None
