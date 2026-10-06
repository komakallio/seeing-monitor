"""The records of the fast seeing path: per-frame metrics and the 60 s seeing window.

The fast-path lane owns this module. Add a field at the end of a class, and keep the rules in
the module documentation of `seeingmon.records.base`.
"""

from __future__ import annotations

from typing import ClassVar, Self

from pydantic import model_validator

from seeingmon.records.base import Record, Storage, quantity

SEEING_WINDOW_FLAGS: dict[str, str] = {
    "degraded": "The system dropped more than 5% of the expected frames.",
    "cloud": "Clouds crossed the star during the window.",
    "twilight": (
        "The Sun was between 0 and 18 degrees below the horizon, so sky light can skew the data."
    ),
    "vibration": "The spectrum shows vibration lines, so the image motion can read high.",
    "saturated": "The star saturated in enough frames to bias the centroid and the width.",
    "partial": "The window ended early, for example because the star neared the edge of the ROI.",
    "time_invalid": "The clock was not synchronized, so `t_utc_ns` is not trustworthy.",
    "heater_on": "The dew heater was on, so heater plumes can add turbulence.",
    "daylight": (
        "The Sun was above the horizon, so a telescope that the Sun heats can add turbulence of "
        "its own."
    ),
    "noisy": (
        "The centroid noise was so large against the image motion that an error of its model "
        "can bias the seeing by more than the configured limit (5% by default)."
    ),
}


class FrameRecord(Record):
    """The metrics of one camera frame, stored as one row of a segment file.

    The system writes per-frame metrics to binary segment files of about 10 minutes, not to
    SQLite rows. A segment stores `station_id`, `profile_id`, `provenance`, `revision`,
    `quality`, and `stream_id` once, in its header, and it stores every other field in each
    row. `t_utc_ns` is the UTC time at the middle of the exposure of the first row of the ROI.
    A float that analysis could not compute is NaN in a row and `null` in the record.
    """

    record_type: ClassVar[str] = "frame"
    storage: ClassVar[Storage] = "segment"
    retention_days: ClassVar[int | None] = 7

    stream_id: int = quantity(
        ge=0,
        le=2**32 - 1,
        definition=(
            "The ID of the stream that produced the frame, which changes at every "
            "reconfiguration of the camera."
        ),
    )
    seq: int = quantity(
        dtype="u4", definition="The sequence number of the frame within its stream."
    )
    t_err_us: int = quantity(
        unit="us",
        dtype="u2",
        definition="The 1-sigma error of `t_utc_ns`, in microseconds, saturating at 65,535.",
    )
    cx_px: float | None = quantity(
        unit="px",
        dtype="f4",
        default=None,
        definition=(
            "The x coordinate of the intensity-weighted centroid, in pixels of the readout mode, "
            "in sensor coordinates."
        ),
    )
    cy_px: float | None = quantity(
        unit="px",
        dtype="f4",
        default=None,
        definition=(
            "The y coordinate of the intensity-weighted centroid, in pixels of the readout mode, "
            "in sensor coordinates."
        ),
    )
    width_x_px: float | None = quantity(
        unit="px",
        dtype="f4",
        default=None,
        definition="The second-moment sigma of the star image along x, in pixels.",
    )
    width_y_px: float | None = quantity(
        unit="px",
        dtype="f4",
        default=None,
        definition="The second-moment sigma of the star image along y, in pixels.",
    )
    peak_dn: int = quantity(
        unit="DN",
        dtype="u2",
        definition="The value of the brightest pixel inside the aperture, in digital numbers.",
    )
    flux_e: float | None = quantity(
        unit="e-",
        dtype="f4",
        default=None,
        definition=(
            "The sum of the pixels inside the aperture, minus the background, in electrons."
        ),
    )
    bg_dn: float | None = quantity(
        unit="DN",
        dtype="f4",
        default=None,
        definition="The local background, which is the median of the ROI border, in DN.",
    )
    flags: int = quantity(
        dtype="u2",
        default=0,
        definition=(
            "A bitmask of facts about the frame: bits 0 to 4 copy `seeingmon.frames.FrameFlag` "
            "(1 `time_invalid`, 2 `recovered`, 4 `incomplete`, 8 `simulated`, 16 `replayed`), "
            "bits 5 to 8 are analysis flags (32 `saturated`, 64 `edge`, 128 `no_star`, "
            "256 `hot_pixel`), and bits 9 to 15 are reserved."
        ),
    )
    dropped_before: int = quantity(
        dtype="u2",
        default=0,
        definition=(
            "The number of frames that the system lost immediately before this frame, "
            "saturating at 65,535."
        ),
    )


class SeeingWindowRecord(Record):
    """The seeing statistics of one window of frames, normally 60 seconds long.

    `t_utc_ns` is the start of the window. A window never spans a reconfiguration of the camera,
    so one `stream_id` covers all of its frames. A statistic that the window cannot support is
    `null`, and `quality` says why.
    """

    record_type: ClassVar[str] = "seeing_window"

    duration_s: float = quantity(unit="s", gt=0, definition="The length of the window, in seconds.")
    stream_id: int = quantity(
        ge=0,
        le=2**32 - 1,
        definition="The ID of the stream that produced the frames of the window.",
    )
    readout_mode: str = quantity(
        min_length=1,
        example="bin1",
        definition="The name of the readout mode in the hardware profile, such as `bin1`.",
    )
    exposure_us: int = quantity(
        unit="us", gt=0, definition="The exposure time of each frame, in microseconds."
    )
    gain: int = quantity(
        ge=0, definition="The camera gain setting, in the units of the camera driver."
    )
    n_frames: int = quantity(ge=0, definition="The number of frames that the window analyzed.")
    n_dropped: int = quantity(
        ge=0, definition="The number of frames that the system lost during the window."
    )
    valid_fraction: float = quantity(
        ge=0,
        le=1,
        definition=(
            "The share of the expected frames that arrived and passed analysis, from 0 to 1."
        ),
    )
    frame_rate_hz: float | None = quantity(
        unit="Hz",
        gt=0,
        default=None,
        definition="The mean frame rate during the window, in hertz.",
    )
    image_motion_rms_x_arcsec: float | None = quantity(
        unit="arcsec",
        ge=0,
        default=None,
        definition=(
            "The RMS of the detrended centroid along sensor x, with the centroid noise "
            "subtracted, in arcseconds."
        ),
    )
    image_motion_rms_y_arcsec: float | None = quantity(
        unit="arcsec",
        ge=0,
        default=None,
        definition=(
            "The RMS of the detrended centroid along sensor y, with the centroid noise "
            "subtracted, in arcseconds."
        ),
    )
    seeing_fwhm_arcsec: float | None = quantity(
        unit="arcsec",
        gt=0,
        default=None,
        definition=(
            "The Kolmogorov seeing FWHM at 500 nm and at the zenith, from the tilt variance with "
            "the outer-scale and exposure corrections, in arcseconds."
        ),
    )
    r0_cm: float | None = quantity(
        unit="cm",
        gt=0,
        default=None,
        definition="The Fried parameter at 500 nm and at the zenith, in centimeters.",
    )
    seeing_fwhm_structure_arcsec: float | None = quantity(
        unit="arcsec",
        gt=0,
        default=None,
        definition=(
            "The seeing FWHM from the structure function of the centroid, which ignores slow "
            "drift and cross-checks `seeing_fwhm_arcsec`, in arcseconds."
        ),
    )
    r0_structure_cm: float | None = quantity(
        unit="cm",
        gt=0,
        default=None,
        definition=(
            "The Fried parameter from the structure function of the centroid, in centimeters."
        ),
    )
    scintillation_index: float | None = quantity(
        default=None,
        definition=(
            "The normalized variance of the flux, minus the Poisson floor, for the exposure "
            "time of the window."
        ),
    )
    centroid_noise_px: float | None = quantity(
        unit="px",
        ge=0,
        default=None,
        definition=(
            "The centroid noise per axis that the estimator subtracted from the variance, in "
            "pixels."
        ),
    )
    width_fwhm_arcsec: float | None = quantity(
        unit="arcsec",
        ge=0,
        default=None,
        definition="The mean FWHM of the star image, from the second-moment width, in arcseconds.",
    )
    peak_mean_dn: float | None = quantity(
        unit="DN",
        ge=0,
        default=None,
        definition="The mean value of the brightest pixel, in digital numbers.",
    )
    flux_mean_e: float | None = quantity(
        unit="e-",
        default=None,
        definition="The mean flux of the star, minus the background, in electrons.",
    )
    background_mean_dn: float | None = quantity(
        unit="DN",
        ge=0,
        default=None,
        definition="The mean local background, in digital numbers.",
    )
    saturated_fraction: float | None = quantity(
        ge=0,
        le=1,
        default=None,
        definition=(
            "The share of the frames whose brightest pixel reaches 98% of full scale, from 0 to 1."
        ),
    )
    outer_scale_m: float | None = quantity(
        unit="m",
        gt=0,
        default=None,
        definition="The outer scale L0 that the estimator assumed, in meters.",
    )
    assumed_wind_ms: float | None = quantity(
        unit="m/s",
        ge=0,
        default=None,
        definition=(
            "The wind speed that the estimator assumed for the exposure correction, in meters "
            "per second."
        ),
    )
    exposure_correction_factor: float | None = quantity(
        gt=0,
        default=None,
        definition=(
            "The factor by which the estimator multiplied the variance to correct for exposure "
            "averaging."
        ),
    )
    outer_scale_correction_factor: float | None = quantity(
        gt=0,
        default=None,
        definition=(
            "The factor by which the estimator multiplied the variance to correct for the "
            "finite outer scale."
        ),
    )
    zenith_angle_deg: float | None = quantity(
        unit="deg",
        ge=0,
        le=180,
        default=None,
        definition="The zenith angle of the star at the middle of the window, in degrees.",
    )
    motion_psd_freq_hz: list[float] | None = quantity(
        unit="Hz",
        default=None,
        definition="The frequencies of the bins of the image-motion spectrum, in hertz.",
    )
    motion_psd_x_arcsec2_per_hz: list[float] | None = quantity(
        unit="arcsec^2/Hz",
        default=None,
        definition=(
            "The power spectral density of the centroid along sensor x in each bin, in square "
            "arcseconds per hertz."
        ),
    )
    motion_psd_y_arcsec2_per_hz: list[float] | None = quantity(
        unit="arcsec^2/Hz",
        default=None,
        definition=(
            "The power spectral density of the centroid along sensor y in each bin, in square "
            "arcseconds per hertz."
        ),
    )
    vibration_lines_hz: list[float] | None = quantity(
        unit="Hz",
        default=None,
        definition=(
            "The frequencies of the spectral lines that the estimator flagged as vibration, in "
            "hertz, or an empty list when it found none."
        ),
    )
    heater_duty: float | None = quantity(
        ge=0,
        le=1,
        default=None,
        definition=(
            "The mean duty cycle of the dew heater during the window, from 0 (off) to 1 (always "
            "on)."
        ),
    )
    sensor_temperature_c: float | None = quantity(
        unit="degC",
        default=None,
        definition="The mean sensor temperature during the window, in degrees Celsius.",
    )
    flags: list[str] = quantity(
        default_factory=list,
        codes=SEEING_WINDOW_FLAGS,
        definition="The conditions that apply to the window, as documented codes.",
    )
    detrend_correction_factor: float | None = quantity(
        gt=0,
        default=None,
        definition=(
            "The factor by which the estimator multiplied the variance to add back the turbulence "
            "that the polynomial detrend removed."
        ),
    )
    centroid_gain_correction_factor: float | None = quantity(
        gt=0,
        default=None,
        definition=(
            "The factor by which the estimator multiplied the variance to correct for the "
            "difference between the centroid in the finite aperture and the G-tilt."
        ),
    )
    motion_psd_dof: float | None = quantity(
        gt=0,
        default=None,
        definition=(
            "The degrees of freedom of each bin of the full-resolution image-motion spectrum, "
            "which sets the relative scatter of a bin to the square root of 2 divided by this "
            "number."
        ),
    )
    background_fraction: float | None = quantity(
        ge=0,
        le=1,
        default=None,
        definition=(
            "The mean local background as a share of the saturation level of the readout mode "
            "and gain, from 0 to 1, with the offset of the camera counted as background."
        ),
    )
    star_snr: float | None = quantity(
        ge=0,
        default=None,
        definition=(
            "The median signal-to-noise ratio of the star in the centroid aperture over the "
            "frames with a usable centroid, which tells the noise of the centroids: the aperture "
            "flux over the root of its photon noise and of the aperture area times the variance "
            "of one pixel, measured on the ROI border (in a bright sky, the SNR of the matched "
            "filter that decides whether the star is there is several times higher)."
        ),
    )

    @model_validator(mode="after")
    def _check_spectrum_lengths(self) -> Self:
        count = None if self.motion_psd_freq_hz is None else len(self.motion_psd_freq_hz)
        for name in ("motion_psd_x_arcsec2_per_hz", "motion_psd_y_arcsec2_per_hz"):
            values = getattr(self, name)
            if values is not None and len(values) != count:
                raise ValueError(f"{name} needs one value for each item of motion_psd_freq_hz")
        return self
