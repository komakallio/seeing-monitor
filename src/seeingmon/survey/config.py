"""The `[survey]` configuration section.

The defaults live in `config/default.d/survey.toml`, and your installation overrides them in
`local/config.toml`. Paths, which belong to one installation, stay empty in the defaults. Every
other default suits the reference camera in bin2 and comes from the research notes. They are
provisional until commissioning shows what a real sky needs (phase 3).
"""

from __future__ import annotations

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

    validity_s: float = 43_200.0  # a solution this old no longer predicts Polaris
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
    max_stars: int = 600  # the brightest stars go to the solver
    cross_check_every: int = 0  # every Nth solved frame also runs the second solver (0: never)
    cross_check_max_px: float = 1.5  # a larger disagreement is reported in the provenance


class CloudConfig(SectionModel):
    """The cloud fraction: the share of expected catalog stars that detection missed."""

    expected_snr: float = 20.0  # a catalog star counts as expected above this clear-sky SNR
    mag_limit: float = 12.0  # and no fainter than this G magnitude
    edge_px: float = 20.0  # catalog stars this close to the frame edge do not count
    min_expected: int = 8  # fewer expected stars give no cloud fraction
    match_radius_px: float = 3.0


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
    min_exposure_s: float = 5.0  # a shorter frame gets no sky quality record (and no extra cost)


class TransparencyConfig(SectionModel):
    """The reference zero point, transparency, and the cloud flag (`survey.transparency`)."""

    window_days: float = 60.0
    quantile: float = 0.9  # the reference is this quantile of the clear zero points
    min_samples: int = 20  # a shorter history gives no reference
    min_stars: int = 12
    max_rms_mag: float = 0.15
    max_cloud_fraction: float = 0.2  # a frame with more clouds does not set the reference
    cloud_flag_fraction: float = 0.3  # the `cloud` flag
    transparency_flag: float = 0.6


class StarEpochConfig(SectionModel):
    """The nightly star summary (`seeingmon.survey.star_epoch`)."""

    min_frames: int = 3  # a star needs this many frames to appear in the summary
    min_snr: float = 20.0


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
    photometry: PhotometryConfig = PhotometryConfig()
    zero_point: ZeroPointConfig = ZeroPointConfig()
    sky: SkyConfig = SkyConfig()
    transparency: TransparencyConfig = TransparencyConfig()
    star_epoch: StarEpochConfig = StarEpochConfig()
