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
from typing import TYPE_CHECKING, Final

from seeingmon.frames import Roi
from seeingmon.profile.derived import SNAPSHOT_OVERHEAD_FLOOR_S

if TYPE_CHECKING:
    from seeingmon.profile.models import Profile, ReadoutMode

ARCSEC_PER_RAD: Final = 206_264.80624709636


@dataclass(frozen=True, slots=True)
class GainPoint:
    """One row of a gain table above gain 0.

    `step` marks a jump: the values change at this gain without a ramp from the row before,
    which must sit at the gain just below.
    """

    gain: int
    e_per_adu: float
    read_noise_e: float
    step: bool = False


@dataclass(frozen=True, slots=True)
class SensorAtGain:
    """The sensor numbers at one gain setting."""

    e_per_adu: float
    read_noise_e: float
    full_well_e: float


# Read from ZWO's charts for the ASI294MM Pro (see "Gain, full well, and read noise" in the
# research notes). The gain 0 values sit in the fields of `SimParams`.
BIN1_GAIN_POINTS: Final = (GainPoint(108, 1.0, 1.8), GainPoint(270, 0.17, 1.38))
BIN2_GAIN_POINTS: Final = (
    GainPoint(119, 1.03, 6.2),
    GainPoint(120, 0.88, 1.85, step=True),
    GainPoint(300, 0.11, 1.3),
)


# ZWO's dark-current chart read off the image, as the reference profile states it (see
# "Reference camera: the non-Pro ASI294MM" in the research notes).
REFERENCE_DARK_TABLE: Final = (
    (-20.0, 0.0022),
    (-10.0, 0.0066),
    (0.0, 0.019),
    (10.0, 0.065),
    (20.0, 0.20),
    (25.0, 0.36),
    (30.0, 0.70),
)


@dataclass(frozen=True, slots=True)
class SimParams:
    """The telescope and one readout mode of the camera.

    A pixel of this mode covers `binning` x `binning` native pixels. The sensor sums them, so
    dark current per pixel scales with `binning` squared. `width` and `height` are the full
    frame in pixels of this mode. `mag0_electron_rate_e_per_s` is the photoelectron rate of
    a magnitude-0 star through an aperture of `mag0_reference_aperture_m`, and the simulator
    scales it by the area of the real aperture. `dark_table` lists `(temperature_c, e/s/px)`
    points. The dark current interpolates them log-linearly in temperature and continues along
    the end segments. With no table, the dark current is `dark_current_e_per_s` at
    `dark_reference_c`, and it doubles every `dark_doubling_c` degrees. `high_speed` holds the
    same mode in the camera's high-speed readout, or `None` when there is none.

    `row_time_s` and `frame_overhead_s` time a video stream. A single exposure (a snapshot) takes
    `snapshot_overhead_s` plus the rows times `snapshot_row_time_s` beyond the exposure, as the
    profile states for the mode. A `None` field means that the profile states none: the video
    row time, and a frame overhead of at least `SNAPSHOT_OVERHEAD_FLOOR_S`.
    """

    mode: str = "bin1"
    width: int = 8288
    height: int = 5644
    pixel_size_um: float = 2.315
    binning: int = 1
    row_time_s: float = 37.6e-6
    frame_overhead_s: float = 7.37e-3
    adc_bits: int = 12
    e_per_adu_gain0: float = 3.5
    read_noise_gain0_e: float = 2.65
    full_well_gain0_e: float = 14_417.0
    gain_points: tuple[GainPoint, ...] = BIN1_GAIN_POINTS
    default_offset: int = 30
    focal_length_m: float = 0.250
    aperture_m: float = 0.050
    wavelength_m: float = 0.60e-6
    mag0_electron_rate_e_per_s: float = 4.6e7
    mag0_reference_aperture_m: float = 0.050
    dark_current_e_per_s: float = 0.2
    dark_reference_c: float = 20.0
    dark_doubling_c: float = 6.0
    dark_table: tuple[tuple[float, float], ...] = ()
    roi_width_multiple: int = 8
    roi_height_multiple: int = 2
    snapshot_overhead_s: float | None = None
    snapshot_row_time_s: float | None = None
    high_speed: SimParams | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.binning < 1:
            raise ValueError("the frame size and the binning must be positive")
        if self.pixel_size_um <= 0 or self.focal_length_m <= 0 or self.aperture_m <= 0:
            raise ValueError("pixel size, focal length, and aperture must be positive")
        if self.wavelength_m <= 0 or self.row_time_s < 0 or self.frame_overhead_s < 0:
            raise ValueError("wavelength and timing must not be negative")
        if (self.snapshot_overhead_s is None) != (self.snapshot_row_time_s is None):
            raise ValueError("snapshot_overhead_s and snapshot_row_time_s go together")
        if (self.snapshot_overhead_s or 0.0) < 0 or (self.snapshot_row_time_s or 0.0) < 0:
            raise ValueError("the snapshot timing must not be negative")
        if not 8 <= self.adc_bits <= 16:
            raise ValueError("adc_bits must be between 8 and 16")
        gains = [point.gain for point in self.gain_points]
        if gains != sorted(set(gains)) or (gains and gains[0] <= 0):
            raise ValueError("gain_points must increase strictly and start above gain 0")
        if len(self.dark_table) == 1:
            raise ValueError("dark_table needs at least two points, or none")
        temperatures = [temperature for temperature, _ in self.dark_table]
        if temperatures != sorted(set(temperatures)):
            raise ValueError("dark_table must increase strictly in temperature")

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
        """The parameters of the normal readout, or of the high-speed readout."""
        if not high_speed:
            return self
        if self.high_speed is None:
            raise ValueError(f"mode {self.mode!r} has no high-speed variant")
        return self.high_speed

    def readout_time_s(self, rows: int) -> float:
        """Time to read `rows` rows: the frame overhead plus the row time for each row."""
        return self.frame_overhead_s + rows * self.row_time_s

    def snapshot_readout_time_s(self, rows: int) -> float:
        """Time that a single exposure of `rows` rows takes beyond the exposure.

        It is the snapshot overhead plus the snapshot row time for each row. A mode without a
        snapshot model takes its video row time and a frame overhead of at least
        `SNAPSHOT_OVERHEAD_FLOOR_S`, as `seeingmon.profile.derived` does.
        """
        if self.snapshot_overhead_s is None or self.snapshot_row_time_s is None:
            overhead_s = max(self.frame_overhead_s, SNAPSHOT_OVERHEAD_FLOOR_S)
            return overhead_s + rows * self.row_time_s
        return self.snapshot_overhead_s + rows * self.snapshot_row_time_s

    # --- sensor ---

    @property
    def adc_full_scale(self) -> int:
        """The largest value that the ADC produces."""
        return (1 << self.adc_bits) - 1

    def sensor_at(self, gain: int) -> SensorAtGain:
        """The electrons per ADU, read noise, and full well at a gain setting.

        The electrons per ADU interpolate log-linearly between the table rows, and the read
        noise interpolates linearly. The full well is the smaller of the gain 0 full well and
        the electrons per ADU times the ADC full scale, because the ADC clips before the well
        fills at high gain. These rules match `seeingmon.profile.derived`.
        """
        if gain < 0:
            raise ValueError("gain must not be negative")
        rows = [
            (0, self.e_per_adu_gain0, self.read_noise_gain0_e, False),
            *((p.gain, p.e_per_adu, p.read_noise_e, p.step) for p in self.gain_points),
        ]
        index = max(i for i, row in enumerate(rows) if row[0] <= gain)
        row_gain, e_per_adu, read_noise, _ = rows[index]
        if index < len(rows) - 1 and not rows[index + 1][3]:
            next_gain, next_e_per_adu, next_read_noise, _ = rows[index + 1]
            fraction = (gain - row_gain) / (next_gain - row_gain)
            e_per_adu = e_per_adu * math.exp(fraction * math.log(next_e_per_adu / e_per_adu))
            read_noise = read_noise + fraction * (next_read_noise - read_noise)
        full_well = min(self.full_well_gain0_e, e_per_adu * self.adc_full_scale)
        return SensorAtGain(e_per_adu, read_noise, full_well)

    def dark_rate_e_per_s(self, temperature_c: float) -> float:
        """Dark current per pixel of this mode at a sensor temperature."""
        if self.dark_table:
            points = self.dark_table
            index = max(
                (i for i in range(len(points) - 1) if points[i][0] <= temperature_c), default=0
            )
            (t_low, rate_low), (t_high, rate_high) = points[index], points[index + 1]
            slope = math.log(rate_high / rate_low) / (t_high - t_low)
            per_native = rate_low * math.exp(slope * (temperature_c - t_low))
        else:
            per_native = self.dark_current_e_per_s * math.pow(
                2.0, (temperature_c - self.dark_reference_c) / self.dark_doubling_c
            )
        return per_native * self.binning**2

    def mag0_rate_e_per_s(self) -> float:
        """Photoelectrons per second from a magnitude-0 star, scaled to this aperture."""
        return (
            self.mag0_electron_rate_e_per_s
            * (self.aperture_m / self.mag0_reference_aperture_m) ** 2
        )

    def sky_rate_e_per_s_px(self, sky_mag_arcsec2: float) -> float:
        """Photoelectrons per second per pixel from a sky of the given surface brightness."""
        per_arcsec2 = self.mag0_rate_e_per_s() * math.pow(10.0, -0.4 * sky_mag_arcsec2)
        return per_arcsec2 * self.plate_scale_arcsec_per_px**2

    # --- construction ---

    @classmethod
    def reference(cls, mode: str = "bin1") -> SimParams:
        """The reference hardware in one readout mode: `bin1` or `bin2`.

        The numbers equal those of `profiles/asi294mm-gs250.toml`, so a test can compare the
        two. They include the high-speed variant of the mode.
        """
        if mode == "bin1":
            normal = cls()
            fast = replace(
                normal,
                adc_bits=10,
                row_time_s=30.0e-6,
                frame_overhead_s=5.88e-3,
                e_per_adu_gain0=3.5 * 4,
                gain_points=tuple(
                    replace(point, e_per_adu=point.e_per_adu * 4) for point in BIN1_GAIN_POINTS
                ),
            )
        elif mode == "bin2":
            normal = cls(
                mode="bin2",
                width=4144,
                height=2822,
                pixel_size_um=4.63,
                binning=2,
                row_time_s=18.5e-6,
                frame_overhead_s=1.22e-3,
                adc_bits=14,
                e_per_adu_gain0=4.05,
                read_noise_gain0_e=8.0,
                full_well_gain0_e=66_387.0,
                gain_points=BIN2_GAIN_POINTS,
                snapshot_overhead_s=0.27,
                snapshot_row_time_s=75.0e-6,
            )
            fast = replace(
                normal,
                adc_bits=12,
                row_time_s=18.5e-6,
                frame_overhead_s=1.22e-3,
                e_per_adu_gain0=4.05 * 4,
                gain_points=tuple(
                    replace(point, e_per_adu=point.e_per_adu * 4) for point in BIN2_GAIN_POINTS
                ),
            )
        else:
            raise ValueError(f"unknown reference mode {mode!r}: use 'bin1' or 'bin2'")
        fast = replace(fast, dark_table=REFERENCE_DARK_TABLE)
        return replace(normal, dark_table=REFERENCE_DARK_TABLE, high_speed=fast)

    @classmethod
    def from_profile(cls, profile: Profile, mode_name: str) -> SimParams:
        """Read the numbers of one readout mode from a hardware profile.

        The result carries the high-speed variant of the mode when the profile defines one.
        The profile gives the dark-current table and the photoelectron rate of a magnitude-0
        star when it has a `photometry` table. Otherwise the defaults of this class apply.
        """
        mode = profile.mode(mode_name)
        normal = cls._from_mode(profile, mode)
        if mode.has_high_speed:
            normal = replace(normal, high_speed=cls._from_mode(profile, mode.high_speed_variant()))
        return normal

    @classmethod
    def modes_from_profile(cls, profile: Profile) -> dict[str, SimParams]:
        """The parameters of every readout mode of a profile, by mode name."""
        return {mode.name: cls.from_profile(profile, mode.name) for mode in profile.readout_modes}

    @classmethod
    def _from_mode(cls, profile: Profile, mode: ReadoutMode) -> SimParams:
        optics = profile.optics
        defaults = cls()
        photometry = profile.photometry
        rate = defaults.mag0_electron_rate_e_per_s
        if photometry is not None and photometry.mag0_electron_rate_e_per_s is not None:
            rate = photometry.mag0_electron_rate_e_per_s
        dark_table: tuple[tuple[float, float], ...] = ()
        if photometry is not None and photometry.dark_current:
            dark_table = tuple((p.temperature_c, p.e_per_s_per_px) for p in photometry.dark_current)
        return cls(
            mode=mode.name,
            width=mode.width_px,
            height=mode.height_px,
            pixel_size_um=mode.pixel_size_um,
            binning=mode.sdk_bin,
            row_time_s=mode.row_time_us * 1e-6,
            frame_overhead_s=mode.frame_overhead_ms * 1e-3,
            adc_bits=mode.adc_bits,
            e_per_adu_gain0=mode.e_per_adu_gain0,
            read_noise_gain0_e=mode.read_noise_gain0_e,
            full_well_gain0_e=mode.full_well_gain0_e,
            gain_points=tuple(
                GainPoint(p.gain, p.e_per_adu, p.read_noise_e, p.step) for p in mode.gain_points
            ),
            focal_length_m=optics.focal_length_mm * 1e-3,
            aperture_m=optics.aperture_mm * 1e-3,
            wavelength_m=optics.wavelength_nm * 1e-9,
            mag0_electron_rate_e_per_s=rate,
            mag0_reference_aperture_m=optics.aperture_mm * 1e-3,
            dark_table=dark_table,
            roi_width_multiple=profile.limits.roi_width_multiple,
            roi_height_multiple=profile.limits.roi_height_multiple,
            snapshot_overhead_s=mode.snapshot_overhead_s,
            snapshot_row_time_s=(
                None if mode.snapshot_row_time_us is None else mode.snapshot_row_time_us * 1e-6
            ),
        )


def reference_modes() -> dict[str, SimParams]:
    """Both readout modes of the reference hardware, by name."""
    return {name: SimParams.reference(name) for name in ("bin1", "bin2")}
