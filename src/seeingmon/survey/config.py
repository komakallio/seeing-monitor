"""The `[survey]` configuration section.

The defaults live in `config/default.d/survey.toml`, and your installation overrides them in
`local/config.toml`. Paths, which belong to one installation, stay empty in the defaults. Every
other default suits the reference camera in bin2 and comes from the research notes. They are
provisional until commissioning shows what a real sky needs (phase 3).
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from seeingmon.config import SectionModel


class DetectConfig(SectionModel):
    """Star detection (`seeingmon.survey.detect.DetectOptions`)."""

    threshold_sigma: float = 5.0  # detection threshold in units of the local background rms
    min_pixels: int = 2
    mesh_px: int = 64  # the background mesh
    edge_margin_px: float = 6.0  # a star nearer to the frame edge gets the `near_edge` flag
    max_stars: int = 3000
    max_saturated_pixels: int = 6  # a star with more is measured by moments, not fitted
    trail_flag_px: float = 1.5
    # 1 searches the frame itself. A larger value searches a copy that sums each block of this many
    # pixels on a side, and fits only the brightest `refine_stars` stars at full resolution. The
    # default 2 passed the comparison against the real catalog (see the architecture).
    coarse_bin: int = 2
    refine_stars: int = 1200


class FitConfig(SectionModel):
    """The pointing fit (`seeingmon.survey.wcs_fit.FitOptions`)."""

    match_radius_px: tuple[float, ...] = (4.0, 2.0, 1.2)  # one match pass for each radius
    clip_sigma: float = 4.0
    min_stars: int = 4
    sigma_floor_px: float = 0.03  # added to each centroid error in quadrature
    final_match_radius_px: float = 1.5  # for the star list and the cloud fraction
    # A fit counts as a solution when it has at least `min_stars` pairs, a residual of at
    # most `max_rms_px`, and either `confident_stars` pairs or `min_match_fraction` of the
    # reliable detections. Random pairs from a wrong prediction fail these tests.
    max_rms_px: float = 1.5
    min_match_fraction: float = 0.3
    confident_stars: int = 30


class PointingConfig(SectionModel):
    """What the tracker and the pointing record do with a solution."""

    # The age at which a solution stops predicting Polaris. 0 sets no limit: a rigid mount keeps
    # its Earth-fixed attitude, so an old solution predicts as well as a new one. A positive value
    # suits a mount that is not rigid.
    validity_s: float = Field(0.0, ge=0, allow_inf_nan=False)
    few_stars: int = 12  # fewer matched stars flag `few_stars`
    moved_arcmin: float = 5.0  # a boresight offset from the reference above this flags `moved`
    moved_roll_deg: float = 0.5  # so does a roll change above this
    tracker_min_stars: int = 8  # a solution with fewer stars does not update the tracker
    tracker_max_rms_px: float = 1.5  # nor does one with a larger residual
    reference_file: str = ""  # the JSON reference solution that commissioning saves


class SolveConfig(SectionModel):
    """Calls to the plate solvers."""

    timeout_s: float = 20.0
    scale_tolerance: float = 0.15  # the solver searches the plate scale +-15%
    hint_radius_deg: float = 2.0  # the search radius around the predicted center
    # The radius around the pole for a first solve and for the retry after the solvers failed near
    # the prediction (a moved mount). 0 searches the whole sky for a first solve and skips the
    # retry, so without an age limit a moved mount never solves again.
    pole_hint_radius_deg: float = 15.0
    max_stars: int = 1000  # the brightest stars go to the solver (an adapter may use fewer)
    cross_check_every: int = 0  # every Nth solved frame also runs the second solver (0: never)
    cross_check_max_px: float = 1.5  # a larger disagreement is reported in the provenance


class CloudConfig(SectionModel):
    """The cloud fraction: the share of expected catalog stars that detection missed."""

    expected_snr: float = 20.0  # a catalog star counts as expected above this clear-sky SNR
    mag_limit: float = 12.0  # and no fainter than this G magnitude
    edge_px: float = 20.0  # catalog stars this close to the frame edge do not count
    min_expected: int = 8  # fewer expected stars give no cloud fraction
    match_radius_px: float = 3.0
    # A catalog star counts as expected only when the search that ran finds it in a clear sky with
    # at least this chance (`seeingmon.survey.completeness`). A long frame whose binned search would
    # miss such stars takes the full search. Provisional.
    min_completeness: float = Field(0.9, gt=0, lt=1, allow_inf_nan=False)


class DarkConfig(SectionModel):
    """The dark library: `seeingmon dark` and `dark_due` (`seeingmon.survey.dark`)."""

    mode: str = "bin2"  # the readout mode and the gain of the survey frames
    gain: int = 120
    exposure_s: float = 30.0  # the survey exposure, which the dark rate divides by
    frames: int = 9
    bias_frames: int = 9
    test_exposure_s: float = 1.0  # the frames that the wait checks
    poll_s: float = 5.0
    stable_polls: int = 2  # dark test frames in a row that count as covered
    wait_timeout_s: float = 1800.0
    max_temperature_spread_c: float = 2.0  # a larger drift during a set gets a warning
    rate_factor: float = 3.0  # a dark frame rises at most this many times the expected rate
    min_rate_e_per_s: float = 0.5  # and at least this much is always allowed
    noise_factor: float = 1.5
    max_tail_fraction: float = 0.001  # the share of pixels far above the median in a dark frame
    hot_sigma: float = 6.0  # a hot pixel rises this many robust sigmas above its neighbors
    hot_min_excess_dn: float = 3.0
    doubling_c: float = 6.0  # the prior doubling temperature of the dark current
    temperature_tolerance_c: float = 3.0  # a set this close to the sensor temperature counts
    max_age_days: float = 183.0  # a set older than this does not count


class FlatConfig(SectionModel):
    """The flat session of the web UI (`seeingmon.survey.flat_session`).

    The session takes the readout mode and the gain from `[survey.dark]`, because the bias of the
    flat comes from the dark library of that mode and gain.
    """

    start_exposure_s: float = 0.02  # where the search for the exposure starts
    max_exposure_s: float = 1.0  # the longest exposure that the search may use
    max_iterations: int = 8  # the most frames that the search takes
    level_tolerance: float = 0.1  # the search stops within this share of the target level
    min_level_fraction: float = 0.25  # a weaker light at the longest exposure is too dim
    drift_percent: float = 3.0  # a frame this far from the median level warns of a drifting light


class PhotometryConfig(SectionModel):
    """Aperture photometry of the matched stars (`seeingmon.survey.photometry`)."""

    aperture_px: float = 5.0  # the radius of the aperture; a trail adds a line of its length
    annulus_inner_px: float = 9.0  # the ring that gives the local background
    annulus_outer_px: float = 14.0
    min_annulus_px: int = 150  # a ring with fewer good pixels gives no measurement
    clip_sigma: float = 3.0
    isolation_flux_ratio: float = 0.02  # a neighbor with this share of the flux spoils a star
    min_snr: float = 20.0  # a star needs this signal-to-noise ratio for the zero point
    growth_aperture_px: float = 12.0  # the wide aperture of the aperture correction
    growth_stars: int = 40  # the brightest stars that measure it; 0 turns the correction off


class ZeroPointConfig(SectionModel):
    """The zero-point fit against Gaia G (`seeingmon.survey.zero_point`)."""

    min_stars: int = 8  # fewer stars give no zero point
    clip_sigma: float = 3.0
    systematic_mag: float = 0.01  # error added to every star: flat field, scintillation, aperture
    color_term_prior: float = 0.0  # the color term when the colors do not span a range
    min_color_std: float = 0.25
    g_min: float = 0.0  # the catalog magnitude range of the stars in the fit
    g_max: float = 13.0


class SkyConfig(SectionModel):
    """The sky brightness and its V equivalent (`seeingmon.survey.sky`)."""

    clip_sigma: float = 3.0
    edge_px: int = 8  # pixels at the frame edge stay out of the sky level
    mask_radius_scale: float = 3.0  # each star hides a disk of this many PSF sigmas
    bp_rp: float = 1.0  # the color that the V conversion assumes for the sky
    sqm_offset_mag: float = 0.0  # the offset that `seeingmon.survey.sqm_fit` gives
    # A shorter frame gets no sky quality record (and no extra cost). Keep it at or below
    # `[survey.twilight] min_exposure_s`, so that every adaptive long frame gets one.
    min_exposure_s: float = 1.0


class TransparencyConfig(SectionModel):
    """The reference zero point, transparency, and the cloud flag (`survey.transparency`)."""

    window_days: float = 365.0  # a long window, so that weeks of haze cannot lower the reference
    quantile: float = 0.95  # the reference is this quantile of the clear zero points
    min_samples: int = 20  # a shorter history gives no reference
    min_stars: int = 12
    max_rms_mag: float = 0.15
    max_cloud_fraction: float = 0.2  # a frame with more clouds does not set the reference
    cloud_flag_fraction: float = 0.3  # the `cloud` flag
    transparency_flag: float = 0.6
    # A frame without a zero point may use the median of this many hours; 0 turns it off.
    fallback_hours: float = 6.0
    # A zero point (mag) that you trust as the clear-sky reference. It replaces the history. 0 turns
    # the pin off.
    pinned_zero_point: float = 0.0


class StarEpochConfig(SectionModel):
    """The nightly star summary (`seeingmon.survey.star_epoch`)."""

    min_frames: int = 3  # a star needs this many frames to appear in the summary
    min_snr: float = 20.0


class TwilightConfig(SectionModel):
    """The survey in a bright sky: the adaptive long exposure and the saturation guard.

    The scheduler reads `target_background_fraction`, `min_exposure_s`, and
    `max_background_fraction` (`seeingmon.scheduler.exposure.SurveyExposure`): the long exposure
    of each survey step puts the sky background at `target_background_fraction` of saturation,
    between `min_exposure_s` and `[scheduler.survey] long_exposure_s`, and the step skips its long
    exposure when even `min_exposure_s` would pass the target. It also skips it while the last
    long frame was beyond `max_background_fraction` at `min_exposure_s`, and the 1 ms frame has
    not darkened since. The pipeline reads the last two keys (`seeingmon.survey.pipeline`): a
    frame over either limit gets the flag `saturated_sky` and no photometry.
    """

    target_background_fraction: float = Field(0.3, gt=0, lt=1, allow_inf_nan=False)
    min_exposure_s: float = Field(1.0, gt=0, allow_inf_nan=False)
    # More saturated pixels than this share of the frame set `saturated_sky`.
    max_saturated_fraction: float = Field(0.01, gt=0, le=1, allow_inf_nan=False)
    # So does a sky background above this share of saturation. It must lie above the target.
    max_background_fraction: float = Field(0.8, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def _guard_is_above_the_target(self) -> Self:
        if self.max_background_fraction <= self.target_background_fraction:
            raise ValueError(
                "max_background_fraction must lie above target_background_fraction, because a long "
                "frame at the target would otherwise get saturated_sky"
            )
        return self


class DarknessConfig(SectionModel):
    """The events `sky.dark` and `sky.clear_verdict` (`seeingmon.services.core.darkness`)."""

    # The sky counts as dark when a line fitted to the sky brightness of the last `frames` solved
    # frames in a row changes by less than this, in magnitudes per hour.
    max_slope_mag_per_hour: float = Field(0.3, gt=0, allow_inf_nan=False)
    frames: int = Field(5, ge=2)
    # A longer time between two frames of the run starts it again, in seconds.
    max_gap_s: float = Field(600.0, gt=0, allow_inf_nan=False)
    # The clear verdict covers this many frames with a cloud fraction after `sky.dark`, solved or
    # not.
    verdict_frames: int = Field(5, ge=1)


class VisibilityConfig(SectionModel):
    """The nightly visibility summary of Polaris (`seeingmon.services.core.visibility`)."""

    # A stretch without watching the sky, longer than this in seconds, between the start of the
    # night and the first detection (or the last detection and the end) censors that detection.
    max_gap_s: float = Field(300.0, ge=0, allow_inf_nan=False)
    # How many `[services.core] health_interval_s` a health record vouches for the station.
    health_span_intervals: float = Field(3.0, ge=1, allow_inf_nan=False)
    # How many nights, the newest that ended included, the first run after a start of core checks
    # for a missing summary.
    catch_up_nights: int = Field(7, ge=1, le=366)


class SurveyConfig(SectionModel):
    """Settings of the survey path. Every key has a default that suits the reference camera."""

    # Files and programs.
    catalog_path: str = ""  # the cap catalog file; empty means "not configured"
    index_dir: str = ""  # the folder with the astrometry.net cap index files
    hot_pixel_file: str = ""  # a NumPy file with the boolean hot-pixel mask of the survey mode
    calibration_dir: str = ""  # the folder with the dark library (in `darks/`); empty: none
    # A flat field of the survey mode, a .npy or FITS image (`seeingmon flat make` or `flat build`).
    flat_file: str = ""  # empty: a unit flat
    night_split_utc_hour: float = 12.0  # the UTC hour that ends a night; pick one in your daytime
    solve_field_command: str = "solve-field"
    astap_command: str = "astap"
    astap_database_dir: str = ""  # the folder with the ASTAP star database; empty uses its default
    solvers: tuple[str, ...] = ("astrometry.net", "astap")  # the order in which to try them
    dut1_s: float = 0.0  # UT1 - UTC in seconds, for the Earth rotation angle

    detect: DetectConfig = DetectConfig()
    fit: FitConfig = FitConfig()
    pointing: PointingConfig = PointingConfig()
    solve: SolveConfig = SolveConfig()
    cloud: CloudConfig = CloudConfig()
    dark: DarkConfig = DarkConfig()
    flat: FlatConfig = FlatConfig()
    photometry: PhotometryConfig = PhotometryConfig()
    zero_point: ZeroPointConfig = ZeroPointConfig()
    sky: SkyConfig = SkyConfig()
    transparency: TransparencyConfig = TransparencyConfig()
    star_epoch: StarEpochConfig = StarEpochConfig()
    twilight: TwilightConfig = TwilightConfig()
    darkness: DarknessConfig = DarknessConfig()
    visibility: VisibilityConfig = VisibilityConfig()
