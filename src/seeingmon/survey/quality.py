"""The sky quality of one survey frame: zero point, sky brightness, transparency, and clouds.

`assess_frame` takes a frame that detection and the pointing step have already processed, and it
makes the `sky_quality` record and the stars for the nightly summary. The steps:

1. **Photometry.** Aperture photometry of the matched, unsaturated, isolated stars
   (`seeingmon.survey.photometry`), in electrons per second.
2. **Zero point.** The fit of the zero point and the color term against Gaia G, with sigma
   clipping (`seeingmon.survey.zero_point`).
3. **Sky.** The dark level for the sensor temperature (`seeingmon.survey.dark`) comes off, the
   flat field divides, the stars and bad pixels hide, and a sigma-clipped median gives the sky
   rate (`seeingmon.survey.sky`). With the zero point, the rate gives the surface brightness
   in the camera band and the V equivalent.
4. **Transparency and clouds.** The zero point against the reference from the clearest conditions
   (`seeingmon.survey.transparency`), the cloud fraction that the pipeline measured with the
   counts of the expected stars and of the ones found, and the limiting magnitude of the frame.
5. **Flags and reasons.** The record carries `cloud`, `dark_due`, and `time_invalid` flags, and a
   `quality` map that says why any value is missing.

A value that the frame cannot support is `None`, and the reason is in `quality`. A frame without
a pointing solution has no matched stars, so it has no zero point, but it can still have a sky
rate and, with a reference zero point, a sky brightness that the reference calibrates. While the
history is too short for a reference, the provisional zero point of the last few hours
(`seeingmon.survey.transparency.provisional_zero_point`) calibrates the sky in its place. The
`quality` map and the `zp_ref_provisional` provenance key say so, and the provisional zero point
sets no transparency.

The `moon` and `twilight` flags need the position of the Sun and the Moon at the site. The survey
path knows no site, so `core` sets them with the scheduler's ephemeris. The `dew` flag needs the
dew point, which only the heater controller knows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from seeingmon.frames import Frame
from seeingmon.profile import Profile
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.survey.catalog import FLAG_TYCHO_ONLY, CapCatalog
from seeingmon.survey.config import SurveyConfig
from seeingmon.survey.dark import DarkModel, DarkStatus
from seeingmon.survey.detect import Detections, StarFlag, star_mask
from seeingmon.survey.geometry import FloatArray
from seeingmon.survey.photometry import (
    PhotometryOptions,
    StarPhotometry,
    measure_matched_stars,
)
from seeingmon.survey.sky import (
    FlatModel,
    SkyMeasurement,
    SkyOptions,
    measure_sky,
    sky_magnitude,
    v_equivalent,
)
from seeingmon.survey.star_epoch import FrameStars
from seeingmon.survey.transparency import (
    LimitingMagnitude,
    TransparencyOptions,
    ZeroPointReference,
    cloud_flag,
    limiting_magnitude,
    min_detectable_flux_e,
    transparency,
)
from seeingmon.survey.wcs_fit import CameraAttitude
from seeingmon.survey.zero_point import (
    ZeroPointFit,
    ZeroPointOptions,
    catalog_errors_mag,
    fit_zero_point,
)

QUALITY_ALGORITHM = "sky-1"

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class QualityOptions:
    """The settings of `assess_frame`, from the `[survey]` configuration."""

    photometry: PhotometryOptions = field(default_factory=PhotometryOptions)
    zero_point: ZeroPointOptions = field(default_factory=ZeroPointOptions)
    sky: SkyOptions = field(default_factory=SkyOptions)
    transparency: TransparencyOptions = field(default_factory=TransparencyOptions)
    min_snr: float = 20.0  # a star needs this signal-to-noise ratio for the zero point
    g_min: float = 0.0  # and a catalog magnitude in this range
    g_max: float = 13.0
    detect_threshold_sigma: float = 5.0  # for the predicted limiting magnitude
    star_mask_scale: float = 3.0
    epoch_min_snr: float = 20.0
    dark_tolerance_c: float = 3.0
    dark_max_age_days: float = 183.0

    @classmethod
    def from_config(cls, config: SurveyConfig) -> QualityOptions:
        photo, zp, sky = config.photometry, config.zero_point, config.sky
        trans = config.transparency
        return cls(
            photometry=PhotometryOptions(
                aperture_px=photo.aperture_px,
                annulus_inner_px=photo.annulus_inner_px,
                annulus_outer_px=photo.annulus_outer_px,
                min_annulus_px=photo.min_annulus_px,
                clip_sigma=photo.clip_sigma,
                isolation_flux_ratio=photo.isolation_flux_ratio,
                growth_aperture_px=photo.growth_aperture_px,
                growth_stars=photo.growth_stars,
            ),
            zero_point=ZeroPointOptions(
                min_stars=zp.min_stars,
                clip_sigma=zp.clip_sigma,
                systematic_mag=zp.systematic_mag,
                color_term_prior=zp.color_term_prior,
                min_color_std=zp.min_color_std,
            ),
            sky=SkyOptions(
                clip_sigma=sky.clip_sigma,
                edge_px=sky.edge_px,
                bp_rp=sky.bp_rp,
                sqm_offset_mag=sky.sqm_offset_mag,
            ),
            transparency=TransparencyOptions(
                window_days=trans.window_days,
                quantile=trans.quantile,
                min_samples=trans.min_samples,
                min_stars=trans.min_stars,
                max_rms_mag=trans.max_rms_mag,
                max_cloud_fraction=trans.max_cloud_fraction,
                cloud_flag_fraction=trans.cloud_flag_fraction,
                transparency_flag=trans.transparency_flag,
                night_split_utc_hour=config.night_split_utc_hour,
                fallback_hours=trans.fallback_hours,
            ),
            min_snr=photo.min_snr,
            g_min=zp.g_min,
            g_max=zp.g_max,
            detect_threshold_sigma=config.detect.threshold_sigma,
            star_mask_scale=sky.mask_radius_scale,
            epoch_min_snr=config.star_epoch.min_snr,
            dark_tolerance_c=config.dark.temperature_tolerance_c,
            dark_max_age_days=config.dark.max_age_days,
        )


@dataclass(frozen=True, slots=True, eq=False)
class FieldStars:
    """The catalog stars that the frame covers, and whether detection found each one.

    The stars lie inside the frame (with a margin) and outside the blobs of saturated stars. A
    star counts as found when a detection matched it, or one lies within the match radius.
    """

    rows: IntArray
    g_mag: FloatArray
    x: FloatArray
    y: FloatArray
    found: BoolArray


@dataclass(frozen=True, slots=True, eq=False)
class SkyQualityResult:
    """The outcome of `assess_frame`.

    `stars` is what the frame adds to the nightly summary: empty when the frame has no zero point
    or is cloudy. The other fields show the intermediate results, for tests and logs.
    """

    record: SkyQualityRecord
    stars: FrameStars
    zero_point: ZeroPointFit | None
    sky: SkyMeasurement | None
    limiting: LimitingMagnitude | None
    n_measured: int


def _python_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _catalog_estimated(catalog: CapCatalog, rows: IntArray) -> BoolArray:
    return np.asarray((catalog.flags[rows] & FLAG_TYCHO_ONLY) != 0, dtype=np.bool_)


def missing_count_reason(field: FieldStars | None, n_expected: int | None) -> str | None:
    """Why the counts behind the cloud fraction are missing, or `None` when the frame has them.

    The counts need the catalog stars of the field, which the solve places (or the stored pointing
    when the frame does not solve), and a clear-sky signal for each star.
    """
    if field is None:
        return "no pointing solution"
    if field.rows.size == 0:
        return "no catalog star lies in the frame"
    if n_expected is None:
        return "no clear-sky signal: no reference zero point and no photometric prior"
    return None


def select_zero_point_stars(
    photometry: StarPhotometry, catalog: CapCatalog, options: QualityOptions
) -> BoolArray:
    """The measured stars that may enter the zero-point fit: enough signal, and a G in range."""
    g = catalog.g_mag[photometry.cat_row]
    return np.asarray(
        (photometry.snr >= options.min_snr) & (g >= options.g_min) & (g <= options.g_max),
        dtype=np.bool_,
    )


def predicted_limit_mag(
    zero_point_mag: float,
    noise_e_px: float,
    exposure_s: float,
    fwhm_px: float,
    trail_px: float,
    threshold_sigma: float,
) -> float | None:
    """The magnitude of a star that a detection at `threshold_sigma` needs, from the noise.

    The aperture follows the star: a disk of 1.5 FWHM in radius, stretched by the trail.
    """
    radius = 1.5 * max(fwhm_px, 0.8)
    area = math.pi * radius**2 + 2.0 * radius * trail_px
    flux = min_detectable_flux_e(threshold_sigma, noise_e_px, area)
    if flux <= 0.0:
        return None
    return zero_point_mag - 2.5 * math.log10(flux / exposure_s)


def assess_frame(
    *,
    station_id: str,
    profile: Profile,
    frame: Frame,
    data: npt.NDArray[np.float32],
    detections: Detections,
    cat_row: IntArray,
    catalog: CapCatalog,
    attitude: CameraAttitude | None,
    field_rows: IntArray,
    field_vectors: FloatArray,
    field: FieldStars | None,
    cloud_fraction: float | None,
    dark_model: DarkModel | None,
    dark_status: DarkStatus | None,
    flat: FlatModel,
    hot_pixels: BoolArray | None,
    zp_reference: ZeroPointReference | None,
    options: QualityOptions,
    provenance: dict[str, str],
    time_invalid: bool,
    n_expected: int | None,
    n_expected_found: int | None,
) -> SkyQualityResult:
    """Make the `sky_quality` record of a frame. See the module documentation.

    `data` is the frame in native counts, `detections` hold sensor positions, and `cat_row` gives
    the catalog row that each detection matched (-1 for none). `field_rows` and `field_vectors`
    are the catalog stars of the field with their apparent places, and `attitude` is the camera
    model of the frame, or `None` when the pointing failed. `hot_pixels` is a mask in the array
    of `data`, and `zp_reference` the reference zero point. While the history is short it is the
    provisional zero point (its `provisional` field is true), or `None` when even that is missing.
    `n_expected` and `n_expected_found` are the counts behind `cloud_fraction`, which the record
    keeps: the expected stars, and those of them that detection found.
    """
    reasons: dict[str, str] = {}
    exposure_s = frame.exposure_us / 1e6
    readout = profile.mode(frame.mode)
    e_per_adu = profile.e_per_adu(frame.mode, frame.gain)
    saturation_dn = profile.saturation(frame.mode, frame.gain).native_dn
    scale = profile.plate_scale_arcsec_per_px(readout)
    roi = frame.roi
    origin = (float(roi.x), float(roi.y))

    # --- The matched stars and the zero point -----------------------------------------------
    fit: ZeroPointFit | None = None
    stars: StarPhotometry | None = None
    chosen = np.zeros(0, dtype=np.bool_)
    if attitude is None or not np.any(cat_row >= 0):
        reasons["zero_point_mag"] = "no pointing solution, so no star matched the catalog"
    else:
        stars = measure_matched_stars(
            data,
            detections,
            cat_row,
            exposure_s=exposure_s,
            e_per_adu=e_per_adu,
            saturation_dn=saturation_dn,
            origin_px=origin,
            options=options.photometry,
            bad=hot_pixels,
            flat_at=flat.at(detections.x, detections.y),
            min_snr=min(options.min_snr, options.epoch_min_snr),
        )
        chosen = select_zero_point_stars(stars, catalog, options)
        g = catalog.g_mag[stars.cat_row[chosen]]
        fit = fit_zero_point(
            g,
            catalog.bp_rp[stars.cat_row[chosen]],
            stars.rate_e_per_s[chosen],
            stars.mag_error[chosen],
            catalog_errors_mag(
                g, _catalog_estimated(catalog, stars.cat_row[chosen]), options.zero_point
            ),
            options.zero_point,
        )
        if fit is None:
            reasons["zero_point_mag"] = (
                f"only {int(chosen.sum())} measurable stars, and the fit needs "
                f"{options.zero_point.min_stars}"
            )
    zp_mag = None if fit is None else fit.zero_point_mag

    # --- The sky ----------------------------------------------------------------------------
    sky: SkyMeasurement | None = None
    if frame.temperature_c is None:
        reasons["sky_mag_arcsec2"] = "the camera reports no sensor temperature"
    elif dark_model is None:
        reasons["sky_mag_arcsec2"] = "no dark model: record a dark set with seeingmon dark"
    else:
        array_stars = detections.shifted(-origin[0], -origin[1])
        mask = star_mask(data.shape, array_stars, radius_scale=options.star_mask_scale)
        sky = measure_sky(
            data,
            star_mask=mask,
            dark_level_dn=dark_model.level_dn(frame.temperature_c, exposure_s),
            flat=flat.image(data.shape, (roi.x, roi.y)),
            exposure_s=exposure_s,
            e_per_adu=e_per_adu,
            scale_arcsec_px=scale,
            saturation_dn=saturation_dn,
            bad=hot_pixels,
            options=options.sky,
        )
        if sky is None:
            reasons["sky_mag_arcsec2"] = "too few clean pixels for a sky level"
        elif sky.rate_e_per_s_arcsec2 <= 0.0:
            reasons["sky_mag_arcsec2"] = "the frame is not above the dark level"

    sky_zero_point = zp_mag
    sky_note = None
    if sky_zero_point is None and sky is not None and zp_reference is not None:
        sky_zero_point = zp_reference.zero_point_mag
        if zp_reference.provisional:
            hours = zp_reference.window_days * 24.0
            sky_note = (
                "calibrated with a provisional zero point "
                f"(the median of {zp_reference.n_samples} frames of the last {hours:g} h), "
                "because the frame has none"
            )
        else:
            sky_note = "calibrated with the reference zero point, because the frame has none"
    sky_mag = (
        None
        if sky is None or sky_zero_point is None
        else sky_magnitude(sky_zero_point, sky.rate_e_per_s_arcsec2)
    )
    if sky is not None and sky.rate_e_per_s_arcsec2 > 0.0 and sky_zero_point is None:
        reasons["sky_mag_arcsec2"] = "no zero point: the frame has none and no reference exists"
    if sky_mag is not None and sky_note is not None:
        reasons["sky_mag_arcsec2"] = sky_note
    color_term = 0.0 if fit is None else fit.color_term
    sky_v = (
        None
        if sky_mag is None
        else v_equivalent(
            sky_mag,
            color_term=color_term,
            bp_rp=options.sky.bp_rp,
            offset_mag=options.sky.sqm_offset_mag,
        )
    )
    if sky_mag is not None and sky_v is None:  # pragma: no cover - the conversion cannot fail
        reasons["sky_mag_arcsec2_v"] = "the conversion failed"

    # --- Transparency, clouds, and the limiting magnitude ------------------------------------
    transparency_value: float | None = None
    if fit is None:
        reasons["transparency"] = "no zero point"
    elif zp_reference is None or zp_reference.provisional:  # a median is no clear-sky level
        reasons["transparency"] = "no reference yet: the history holds too few clear zero points"
    else:
        transparency_value = transparency(fit.zero_point_mag, zp_reference)
    count_reason = missing_count_reason(field, n_expected)
    if count_reason is not None:
        reasons["n_expected"] = reasons["n_expected_found"] = count_reason
    if cloud_fraction is None:
        reasons["cloud_fraction"] = count_reason or f"too few expected stars ({n_expected})"

    limiting: LimitingMagnitude | None = None
    if field is None:
        reasons["limiting_mag"] = "no pointing solution"
    else:
        predicted = None
        limit_zero_point = (
            zp_mag
            if zp_mag is not None
            else (None if zp_reference is None else zp_reference.zero_point_mag)
        )
        if limit_zero_point is not None and sky is not None and len(detections):
            reliable = detections.reliable()
            fwhm = float(np.median(detections.fwhm_px[reliable])) if reliable.any() else 1.1
            trail = float(np.median(detections.trail_length_px))
            predicted = predicted_limit_mag(
                limit_zero_point,
                sky.noise_dn * e_per_adu,
                exposure_s,
                fwhm,
                trail,
                options.detect_threshold_sigma,
            )
        limiting = limiting_magnitude(field.g_mag, field.found, predicted_mag=predicted)
        if limiting.value is None:
            reasons["limiting_mag"] = {
                "too_few_stars": "too few catalog stars in the field",
                "no_detections": "the frame detects fewer than half of even the brightest stars",
                "beyond_catalog": "the frame detects more than half of the stars at the faint "
                "end of the catalog, and the noise gives no prediction",
            }.get(limiting.status, limiting.status)
        elif limiting.status == "predicted":
            reasons["limiting_mag"] = "beyond the catalog: predicted from the noise of the frame"

    # --- Flags ------------------------------------------------------------------------------
    flags: list[str] = []
    if cloud_flag(cloud_fraction, transparency_value, options.transparency):
        flags.append("cloud")
    if dark_status is None or dark_status.due:
        flags.append("dark_due")
    if time_invalid:
        flags.append("time_invalid")

    out_provenance = dict(provenance)
    out_provenance["algo"] = QUALITY_ALGORITHM
    if zp_reference is not None:
        out_provenance["zp_ref"] = f"{zp_reference.zero_point_mag:.4f}"
        out_provenance["zp_ref_n"] = str(zp_reference.n_samples)
        if zp_reference.provisional:
            out_provenance["zp_ref_provisional"] = "true"
    if stars is not None and stars.n_growth_stars:
        out_provenance["ap_corr"] = f"{stars.aperture_correction:.4f}"

    rate = None if sky is None or sky.rate_e_per_s_arcsec2 < 0.0 else sky.rate_e_per_s_arcsec2
    record = SkyQualityRecord(
        station_id=station_id,
        t_utc_ns=frame.t_utc_ns,
        profile_id=profile.id,
        provenance=out_provenance,
        quality=reasons or None,
        sky_mag_arcsec2=_python_float(sky_mag),
        sky_mag_arcsec2_v=_python_float(sky_v),
        zero_point_mag=_python_float(zp_mag),
        zero_point_rms_mag=None if fit is None else float(fit.rms_mag),
        color_term=None if fit is None else float(fit.color_term),
        transparency=_python_float(transparency_value),
        cloud_fraction=_python_float(cloud_fraction),
        limiting_mag=None if limiting is None else _python_float(limiting.value),
        n_stars_used=0 if fit is None else int(fit.n_used),
        sky_rate_e_per_s_arcsec2=_python_float(rate),
        dark_model_version=None if dark_model is None else dark_model.version,
        flags=flags,
        n_expected=n_expected,
        n_expected_found=n_expected_found,
    )

    # --- The stars for the nightly summary --------------------------------------------------
    epoch_stars = FrameStars.empty()
    if (
        fit is not None
        and stars is not None
        and attitude is not None
        and "cloud" not in flags
        and len(stars)
    ):
        epoch_stars = _epoch_stars(
            stars, fit, catalog, attitude, field_rows, field_vectors, scale, options
        )
    return SkyQualityResult(
        record=record,
        stars=epoch_stars,
        zero_point=fit,
        sky=sky,
        limiting=limiting,
        n_measured=0 if stars is None else len(stars),
    )


def _epoch_stars(
    stars: StarPhotometry,
    fit: ZeroPointFit,
    catalog: CapCatalog,
    attitude: CameraAttitude,
    field_rows: IntArray,
    field_vectors: FloatArray,
    scale_arcsec_px: float,
    options: QualityOptions,
) -> FrameStars:
    """The offsets and the magnitudes of the measured stars, for the nightly summary."""
    keep = stars.snr >= options.epoch_min_snr
    rows = stars.cat_row[keep]
    if rows.size == 0 or field_rows.size == 0:
        return FrameStars.empty()
    order = np.argsort(field_rows)
    sorted_rows = field_rows[order]
    position = np.searchsorted(sorted_rows, rows)
    position = np.clip(position, 0, sorted_rows.size - 1)
    known = sorted_rows[position] == rows
    if not known.any():
        return FrameStars.empty()
    vector_index = order[position[known]]
    predicted_x, predicted_y, front = attitude.project(field_vectors[vector_index])
    use = np.flatnonzero(known)
    ok = front
    x = stars.x[keep][use][ok]
    y = stars.y[keep][use][ok]
    dx = (x - predicted_x[ok]) * scale_arcsec_px
    dy = (y - predicted_y[ok]) * scale_arcsec_px
    cat = rows[use][ok]
    rate = stars.rate_e_per_s[keep][use][ok]
    color = catalog.bp_rp[cat]
    color = np.where(np.isfinite(color), color, 0.0)
    mag = fit.zero_point_mag - 2.5 * np.log10(rate) + fit.color_term * color
    return FrameStars.build(cat, dx, dy, mag)


def field_stars(
    catalog: CapCatalog,
    attitude: CameraAttitude,
    rows: IntArray,
    vectors: FloatArray,
    detections: Detections,
    cat_row: IntArray,
    *,
    roi_bounds: tuple[float, float, float, float],
    edge_px: float,
    match_radius_px: float,
) -> FieldStars:
    """The catalog stars of the field that lie inside the frame, with their detection status.

    `roi_bounds` is `(x0, y0, x1, y1)` of the frame in sensor pixels. A star within `edge_px` of
    the edge, or inside the blob of a saturated star, stays out. A star is found when a detection
    matched it, or when a detection lies within `match_radius_px`.
    """
    x, y, front = attitude.project(vectors)
    x0, y0, x1, y1 = roi_bounds
    inside = (
        front
        & (x > x0 + edge_px)
        & (x < x1 - 1 - edge_px)
        & (y > y0 + edge_px)
        & (y < y1 - 1 - edge_px)
    )
    saturated = detections.has(StarFlag.SATURATED)
    if saturated.any():
        reach = 2.0 * np.sqrt(np.maximum(detections.n_pixels[saturated], 1))
        for sx, sy, r in zip(detections.x[saturated], detections.y[saturated], reach, strict=True):
            inside &= np.hypot(x - sx, y - sy) > r
    keep = np.flatnonzero(inside)
    kept_rows = rows[keep]
    detected_rows = {int(row) for row in cat_row[cat_row >= 0]}
    found = np.array([int(row) in detected_rows for row in kept_rows], dtype=np.bool_)
    if (~found).any() and len(detections):
        missing = np.flatnonzero(~found)
        for i in missing:
            distance = float(np.min(np.hypot(detections.x - x[keep][i], detections.y - y[keep][i])))
            found[i] = distance <= match_radius_px
    return FieldStars(
        rows=kept_rows,
        g_mag=catalog.g_mag[kept_rows],
        x=x[keep],
        y=y[keep],
        found=found,
    )
