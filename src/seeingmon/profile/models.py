"""The profile schema: the description of one camera and telescope pair.

A profile is a TOML file in `profiles/` that holds hardware values only: the sensor, the
optics, the readout modes, and the limits. The software derives everything else from it
(plate scale, field of view, ROI sizes, sampling, saturation, and frame rates, in
`seeingmon.profile.derived`), so no module hard-codes a pixel size, a focal length, or a bit
depth.

Field names carry their units. Every model is frozen and rejects unknown keys, so a typo in a
profile file is an error and not a silently ignored value.

**Gain at gain 0.** A readout mode states its electrons per ADU, read noise, and full well at
gain 0 as plain fields. Its `gain_points` list the rows for higher gains, so the gain 0 row
is never stated twice. See `seeingmon.profile.derived` for how the rows interpolate.

**High-speed mode.** A camera can trade ADC bits for speed. A mode states the high-speed ADC
bits, row time, and frame overhead when they differ, and `ReadoutMode.high_speed_variant`
returns the mode as it runs in high-speed mode.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_serializer,
    field_validator,
    model_validator,
)

from seeingmon.frames import MODE_NAME_BYTES, PixelFormat, Roi
from seeingmon.profile import derived
from seeingmon.profile.errors import ProfileError

PROFILE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
SENSOR_SIZE_TOLERANCE = 0.05  # the readout modes of one sensor must agree within 5%

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
AdcBits = Annotated[int, Field(ge=1, le=16)]  # the 16-bit container is the largest


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GainPoint(_Model):
    """One row of a readout mode's gain table, for a gain above 0.

    `step` marks a discontinuity: the values jump between the previous point and this one, as
    they do when high conversion gain switches on. A step point must sit one gain unit above
    the previous point, and the values never interpolate across it.
    """

    gain: PositiveInt
    e_per_adu: Positive
    read_noise_e: Positive
    step: StrictBool = False


class ReadoutMode(_Model):
    """One way the camera reads out the sensor, such as bin1 or bin2.

    `name` is the label that frames carry. `sdk_bin` is the binning factor that the vendor
    SDK takes. `width_px`, `height_px`, and `pixel_size_um` describe the mode after binning.
    The `*_gain0_*` fields hold the values at gain 0, and `gain_points` hold the rows above it.
    """

    name: str
    sdk_bin: PositiveInt
    width_px: PositiveInt
    height_px: PositiveInt
    pixel_size_um: Positive
    adc_bits: AdcBits
    adc_bits_high_speed: AdcBits | None = None
    full_well_gain0_e: Positive
    read_noise_gain0_e: Positive
    e_per_adu_gain0: Positive
    row_time_us: Positive
    frame_overhead_ms: NonNegative
    row_time_us_high_speed: Positive | None = None
    frame_overhead_ms_high_speed: NonNegative | None = None
    gain_points: tuple[GainPoint, ...] = ()

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not value or len(value) > MODE_NAME_BYTES or not all("!" <= ch <= "~" for ch in value):
            raise ValueError(
                f"a mode name is 1 to {MODE_NAME_BYTES} printable ASCII characters "
                f"without spaces, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _check_gain_points(self) -> Self:
        previous = 0
        for index, point in enumerate(self.gain_points):
            if point.gain <= previous:
                raise ValueError(
                    "gain_points must be strictly increasing and start above gain 0 "
                    "(the gain 0 values are the *_gain0_* fields); "
                    f"gain_points[{index}] has gain {point.gain} after gain {previous}"
                )
            if point.step and point.gain != previous + 1:
                raise ValueError(
                    f"gain_points[{index}] marks a step at gain {point.gain}, so the previous "
                    f"point must be at gain {point.gain - 1}, not {previous}"
                )
            previous = point.gain
        return self

    @property
    def has_high_speed(self) -> bool:
        """Whether the mode states anything that differs in high-speed mode."""
        return (
            self.adc_bits_high_speed is not None
            or self.row_time_us_high_speed is not None
            or self.frame_overhead_ms_high_speed is not None
        )

    def high_speed_variant(self) -> ReadoutMode:
        """The mode as it runs in the camera's high-speed mode.

        The variant takes the high-speed ADC bits, row time, and frame overhead where the mode
        states them. With fewer ADC bits, electrons per ADU grows by 2^(bits lost), because the
        ADC covers the same analog range with fewer steps. That scaling is an assumption: the
        vendor does not publish the high-speed gain charts. Raises `ProfileError` when the mode
        has no high-speed variant.
        """
        if not self.has_high_speed:
            raise ProfileError(f"readout mode {self.name!r} has no high-speed variant")
        bits = self.adc_bits if self.adc_bits_high_speed is None else self.adc_bits_high_speed
        scale = 2.0 ** (self.adc_bits - bits)
        return self.model_copy(
            update={
                "adc_bits": bits,
                "adc_bits_high_speed": None,
                "e_per_adu_gain0": self.e_per_adu_gain0 * scale,
                "gain_points": tuple(
                    point.model_copy(update={"e_per_adu": point.e_per_adu * scale})
                    for point in self.gain_points
                ),
                "row_time_us": self.row_time_us
                if self.row_time_us_high_speed is None
                else self.row_time_us_high_speed,
                "frame_overhead_ms": self.frame_overhead_ms
                if self.frame_overhead_ms_high_speed is None
                else self.frame_overhead_ms_high_speed,
                "row_time_us_high_speed": None,
                "frame_overhead_ms_high_speed": None,
            }
        )


class Sensor(_Model):
    """The camera: its name, whether it cools the sensor, and whether it reports a temperature."""

    name: Annotated[str, Field(min_length=1)]
    cooled: StrictBool
    has_temperature_sensor: StrictBool


class Optics(_Model):
    """The telescope: focal length, aperture, effective wavelength, and usable image circle.

    `image_circle_diameter_mm` is the diameter of the circle on the sensor where the image is
    good. Leave it out when the full sensor is usable.
    """

    focal_length_mm: Positive
    aperture_mm: Positive
    wavelength_nm: Positive
    image_circle_diameter_mm: Positive | None = None

    @property
    def f_number(self) -> float:
        """The focal ratio: focal length over aperture."""
        return self.focal_length_mm / self.aperture_mm


class Limits(_Model):
    """What the camera and its SDK accept.

    The ROI rules count binned pixels: the width is a multiple of `roi_width_multiple` and the
    height is a multiple of `roi_height_multiple`. Each range is (minimum, maximum). The offset
    range is optional because vendors do not always publish it.
    """

    roi_width_multiple: PositiveInt
    roi_height_multiple: PositiveInt
    gain_range: tuple[NonNegativeInt, NonNegativeInt]
    exposure_us_range: tuple[PositiveInt, PositiveInt]
    offset_range: tuple[NonNegativeInt, NonNegativeInt] | None = None

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        ranges: list[tuple[str, tuple[int, int] | None]] = [
            ("gain_range", self.gain_range),
            ("exposure_us_range", self.exposure_us_range),
            ("offset_range", self.offset_range),
        ]
        for name, bounds in ranges:
            if bounds is not None and bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be (minimum, maximum), got {list(bounds)}")
        return self


class ModeSelection(_Model):
    """A choice of readout mode for one use, with an optional default pixel format."""

    mode: str
    pixel_format: PixelFormat | None = None

    @field_validator("pixel_format", mode="before")
    @classmethod
    def _parse_pixel_format(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return PixelFormat[value.upper()]
            except KeyError:
                names = ", ".join(fmt.name for fmt in PixelFormat)
                raise ValueError(f"pixel_format must be one of {names}, got {value!r}") from None
        return value

    @field_serializer("pixel_format")
    def _serialize_pixel_format(self, value: PixelFormat | None) -> str | None:
        return None if value is None else value.name


class DarkCurrentPoint(_Model):
    """One point of the dark-current prior: electrons per second per pixel at a temperature."""

    temperature_c: Finite
    e_per_s_per_px: Positive


class Photometry(_Model):
    """Optional photometric priors.

    `mag0_electron_rate_e_per_s` is the photoelectron rate of a magnitude-0 star in the
    unfiltered camera. It is an estimate, and `mag0_electron_rate_rel_uncertainty` states how
    good it is as a fraction (0.3 means plus or minus 30%). The two go together. The
    `dark_current` table lists points in increasing temperature order.
    """

    mag0_electron_rate_e_per_s: Positive | None = None
    mag0_electron_rate_rel_uncertainty: Annotated[float, Field(gt=0, le=1)] | None = None
    dark_current: tuple[DarkCurrentPoint, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.mag0_electron_rate_e_per_s is None) != (
            self.mag0_electron_rate_rel_uncertainty is None
        ):
            raise ValueError(
                "photometry: mag0_electron_rate_e_per_s and mag0_electron_rate_rel_uncertainty "
                "go together, so state how well you know the rate (0.3 means plus or minus 30%)"
            )
        if len(self.dark_current) == 1:
            raise ValueError("photometry: dark_current needs at least two points, or none")
        for earlier, later in zip(self.dark_current, self.dark_current[1:], strict=False):
            if later.temperature_c <= earlier.temperature_c:
                raise ValueError(
                    "photometry: dark_current must be in strictly increasing temperature order; "
                    f"{later.temperature_c} follows {earlier.temperature_c}"
                )
        return self


class Profile(_Model):
    """One hardware configuration: sensor, optics, readout modes, and limits.

    `id` must equal the file name without `.toml`. `fast_mode` and `survey_mode` each name a
    readout mode. The methods below return the derived values of `seeingmon.profile.derived`;
    each takes a readout mode by name or as a `ReadoutMode`.
    """

    id: Annotated[str, Field(pattern=PROFILE_ID_PATTERN, max_length=64)]
    description: Annotated[str, Field(min_length=1)]
    sensor: Sensor
    optics: Optics
    limits: Limits
    readout_modes: Annotated[tuple[ReadoutMode, ...], Field(min_length=1)]
    fast_mode: ModeSelection
    survey_mode: ModeSelection
    photometry: Photometry | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        names = [mode.name for mode in self.readout_modes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"readout mode names must be unique; repeated: {', '.join(duplicates)}"
            )
        for label, selection in (("fast_mode", self.fast_mode), ("survey_mode", self.survey_mode)):
            if selection.mode not in names:
                raise ValueError(
                    f"{label}: unknown readout mode {selection.mode!r}; "
                    f"this profile defines {', '.join(names)}"
                )
        for mode in self.readout_modes:
            if (
                mode.width_px < self.limits.roi_width_multiple
                or mode.height_px < self.limits.roi_height_multiple
            ):
                raise ValueError(
                    f"readout mode {mode.name!r} is {mode.width_px} x {mode.height_px} pixels, "
                    "smaller than one ROI step "
                    f"({self.limits.roi_width_multiple} x {self.limits.roi_height_multiple})"
                )
            if mode.gain_points and mode.gain_points[-1].gain > self.limits.gain_range[1]:
                raise ValueError(
                    f"readout mode {mode.name!r} has a gain point at {mode.gain_points[-1].gain}, "
                    f"above the gain range {list(self.limits.gain_range)}"
                )
        first = self.readout_modes[0]
        first_size = derived.sensor_size_mm(first)
        for mode in self.readout_modes[1:]:
            size = derived.sensor_size_mm(mode)
            if any(
                abs(a - b) > SENSOR_SIZE_TOLERANCE * a
                for a, b in zip(first_size, size, strict=True)
            ):
                raise ValueError(
                    f"readout modes {first.name!r} and {mode.name!r} cover different sensor "
                    f"areas ({first_size[0]:.2f} x {first_size[1]:.2f} mm and "
                    f"{size[0]:.2f} x {size[1]:.2f} mm); check the resolution and the pixel size"
                )
        return self

    # --- Readout modes -------------------------------------------------------------------

    def mode(self, name: str, *, high_speed: bool = False) -> ReadoutMode:
        """The readout mode called `name`, or its high-speed variant.

        Raises `ProfileError` for an unknown name, or when `high_speed` is set and the mode has
        no high-speed variant.
        """
        for candidate in self.readout_modes:
            if candidate.name == name:
                return candidate.high_speed_variant() if high_speed else candidate
        known = ", ".join(mode.name for mode in self.readout_modes)
        raise ProfileError(f"unknown readout mode {name!r}; this profile defines {known}")

    @property
    def fast_readout(self) -> ReadoutMode:
        """The readout mode that `fast_mode` names."""
        return self.mode(self.fast_mode.mode)

    @property
    def survey_readout(self) -> ReadoutMode:
        """The readout mode that `survey_mode` names."""
        return self.mode(self.survey_mode.mode)

    def _resolve(self, mode: str | ReadoutMode) -> ReadoutMode:
        return self.mode(mode) if isinstance(mode, str) else mode

    # --- Derived values ------------------------------------------------------------------

    def plate_scale_arcsec_per_px(self, mode: str | ReadoutMode) -> float:
        """The plate scale of a readout mode in arcsec per pixel."""
        return derived.plate_scale_arcsec_per_px(self._resolve(mode), self.optics)

    def field_of_view(self, mode: str | ReadoutMode) -> derived.FieldOfView:
        """The field of view of a readout mode, in degrees."""
        return derived.field_of_view(self._resolve(mode), self.optics)

    def roi_size_px(self, mode: str | ReadoutMode, full_width_arcmin: float) -> tuple[int, int]:
        """The ROI size (width, height) in pixels for a square patch of sky."""
        return derived.roi_size_px(self._resolve(mode), self.optics, self.limits, full_width_arcmin)

    def clamp_roi(
        self, mode: str | ReadoutMode, x: float, y: float, width: float, height: float
    ) -> Roi:
        """Round and clamp a requested ROI to the ROI rules and the frame."""
        return derived.clamp_roi(self._resolve(mode), self.limits, x, y, width, height)

    def airy_fwhm_arcsec(self, wavelength_nm: float | None = None) -> float:
        """The Airy FWHM in arcsec at the given wavelength (default: the effective one)."""
        return derived.airy_fwhm_arcsec(self.optics, wavelength_nm)

    def airy_fwhm_px(self, mode: str | ReadoutMode, wavelength_nm: float | None = None) -> float:
        """The Airy FWHM in pixels of a readout mode."""
        return derived.airy_fwhm_px(self._resolve(mode), self.optics, wavelength_nm)

    def sampling_ratio(self, mode: str | ReadoutMode, wavelength_nm: float | None = None) -> float:
        """The pixel size over lambda times the f-number. Below 1 means no centroid phase bias."""
        return derived.sampling_ratio(self._resolve(mode), self.optics, wavelength_nm)

    def e_per_adu(self, mode: str | ReadoutMode, gain: float) -> float:
        """The electrons per ADU (native counts) at a gain."""
        return derived.e_per_adu(self._resolve(mode), gain)

    def read_noise_e(self, mode: str | ReadoutMode, gain: float) -> float:
        """The read noise in electrons at a gain."""
        return derived.read_noise_e(self._resolve(mode), gain)

    def full_well_e(self, mode: str | ReadoutMode, gain: float) -> float:
        """The effective full well in electrons at a gain."""
        return derived.full_well_e(self._resolve(mode), gain)

    def saturation(self, mode: str | ReadoutMode, gain: float) -> derived.Saturation:
        """The saturation level at a gain, in native ADC counts and 16-bit container counts."""
        return derived.saturation(self._resolve(mode), gain)

    def row_time_ns(self, mode: str | ReadoutMode) -> int:
        """The row time of a readout mode in nanoseconds."""
        return derived.row_time_ns(self._resolve(mode))

    def frame_period_s(
        self, mode: str | ReadoutMode, roi_height_px: int, exposure_us: float
    ) -> float:
        """The time between frames for an ROI height and an exposure."""
        return derived.frame_period_s(self._resolve(mode), roi_height_px, exposure_us)

    def max_frame_rate_hz(
        self, mode: str | ReadoutMode, roi_height_px: int, exposure_us: float
    ) -> float:
        """The highest frame rate for an ROI height and an exposure."""
        return derived.max_frame_rate_hz(self._resolve(mode), roi_height_px, exposure_us)

    def data_rate_bytes_per_s(
        self,
        mode: str | ReadoutMode,
        roi_width_px: int,
        roi_height_px: int,
        exposure_us: float,
        pixel_format: PixelFormat = PixelFormat.RAW16,
    ) -> float:
        """The pixel data rate in bytes per second at the highest frame rate."""
        return derived.data_rate_bytes_per_s(
            self._resolve(mode), roi_width_px, roi_height_px, exposure_us, pixel_format
        )

    def star_electron_rate_e_per_s(self, magnitude: float) -> float:
        """The photoelectron rate of a star of the given magnitude (an estimate)."""
        return derived.star_electron_rate_e_per_s(self.photometry, magnitude)

    def dark_current_e_per_s_per_px(self, temperature_c: float) -> float:
        """The dark-current prior in electrons per second per pixel at a sensor temperature."""
        return derived.dark_current_e_per_s_per_px(self.photometry, temperature_c)
