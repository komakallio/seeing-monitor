"""A synthetic sky for the survey tests: catalog stars rendered through a known pointing.

The renderer takes a catalog, a camera attitude fixed to the Earth, and a time. It computes the
apparent places of the stars, turns the sky through the exposure (so far stars trail), and
draws each position as a Gaussian PSF that it integrates over every pixel. It adds a sky level,
hot pixels, a halo around the brightest stars, Poisson and read noise, the camera offset, and
saturation, and it packs the result into a `Frame` as the driver would, with the ADC value in
the high bits of 16.

The helpers draw the stars with their own code (`scipy.special.erf`), so a test that fits the
frame does not share the model code with the fit. They do share `seeingmon.survey.apparent`,
which `test_apparent` validates against `astropy`, and `astropy_apparent_vectors` gives the
tests a way to render with `astropy` as the truth.

**The site.** The pointing code needs no site, and these tests carry none. A frame needs only
a time, because the Earth rotation angle turns the sky.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.frames import Frame, FrameFlag, Roi, TimeQuality
from seeingmon.profile import Profile, load_profile
from seeingmon.solvers.base import SolveRequest, SolveResult
from seeingmon.survey import apparent
from seeingmon.survey._scipy import erf
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.catalog_build import propagate
from seeingmon.survey.geometry import (
    ARCSEC_PER_RAD,
    FloatArray,
    nearest_rotation,
    radec_to_vector,
    rot_z,
    tangent_basis,
    vector_to_radec,
)

# A clear autumn night. The date fixes the aberration and the Earth rotation angle.
NIGHT_UTC_NS = iso_to_utc_ns("2026-10-01T22:00:00Z")
REFERENCE_PROFILE_ID = "asi294mm-gs250"

TransmissionFunction = Callable[[FloatArray, FloatArray], FloatArray]


def reference_profile() -> Profile:
    return load_profile(REFERENCE_PROFILE_ID)


def cropped_profile(width: int, height: int) -> Profile:
    """The reference profile with a smaller sensor: `width` x `height` in bin2, twice that in bin1.

    The pixel size and the focal length stay, so the plate scale is the same and the field
    shrinks. Small frames keep the tests fast.
    """
    data = reference_profile().model_dump(mode="python")
    for mode in data["readout_modes"]:
        factor = 2 // mode["sdk_bin"]
        mode["width_px"] = width * factor
        mode["height_px"] = height * factor
    return Profile.model_validate(data)


def make_attitude(polar_distance_deg: float, azimuth_deg: float, roll_deg: float) -> FloatArray:
    """A camera attitude fixed to the Earth: `camera = R @ earth_fixed`.

    The boresight lies `polar_distance_deg` from the pole, at the azimuth `azimuth_deg` about
    it. The roll is the position angle of the direction to the pole in the image, measured from
    image up toward image left (`roll_deg = 0` puts the pole straight up). The camera axes are
    x to the right, y down, and z along the boresight, as `seeingmon.survey.wcs_fit` defines.
    """
    theta = np.radians(polar_distance_deg)
    lam = np.radians(azimuth_deg)
    boresight = np.array([np.sin(theta) * np.cos(lam), np.sin(theta) * np.sin(lam), np.cos(theta)])
    pole = np.array([0.0, 0.0, 1.0])
    toward_pole = pole - boresight * (pole @ boresight)
    length = np.linalg.norm(toward_pole)
    if length < 1e-12:  # on the pole itself, use the limit along the meridian at this azimuth
        toward_pole = np.array([-np.cos(lam), -np.sin(lam), 0.0])
    else:
        toward_pole = toward_pole / length
    down = -toward_pole  # with no roll, the pole is up, so down points away from it
    right = np.cross(down, boresight)
    rho = np.radians(roll_deg)
    ex = np.cos(rho) * right + np.sin(rho) * down
    ey = -np.sin(rho) * right + np.cos(rho) * down
    return nearest_rotation(np.stack([ex, ey, boresight]))  # exact near the pole too


def prior_zero_point(profile: Profile) -> float:
    """The zero point of the profile's photometric prior: the magnitude with 1 e-/s."""
    photometry = profile.photometry
    assert photometry is not None
    assert photometry.mag0_electron_rate_e_per_s is not None
    return float(2.5 * np.log10(photometry.mag0_electron_rate_e_per_s))


def synthetic_catalog(
    *,
    cap_radius_deg: float = 6.0,
    density_scale: float = 1.0,
    seed: int = 0,
    faint_limit: float = 13.0,
    bright_limit: float = 5.0,
    with_polaris: bool = True,
) -> CapCatalog:
    """Made-up stars in a cap around the pole, with the star counts of the real catalog.

    The real catalog holds 82,055 stars to G = 13 within 15 degrees (116 per square degree),
    and the counts grow by a factor of 2.2 each magnitude. Positions are uniform on the sphere.
    `density_scale` multiplies the number of stars. The catalog also holds Polaris, with its
    real astrometry (the only real star here), so a test can saturate or remove it.
    """
    rng = np.random.default_rng(seed)
    area_deg2 = 2.0 * np.pi * (1.0 - np.cos(np.radians(cap_radius_deg))) * (180.0 / np.pi) ** 2
    count = round(116.0 * area_deg2 * density_scale)
    cos_polar = rng.uniform(np.cos(np.radians(cap_radius_deg)), 1.0, count)
    dec = 90.0 - np.degrees(np.arccos(cos_polar))
    ra = rng.uniform(0.0, 360.0, count)
    slope = np.log10(2.2)
    low, high = 10 ** (slope * bright_limit), 10 ** (slope * faint_limit)
    g_mag = np.log10(rng.uniform(low, high, count)) / slope
    pm_ra = rng.normal(0.0, 8.0, count)
    pm_dec = rng.normal(0.0, 8.0, count)
    parallax = rng.uniform(0.2, 4.0, count)
    bp_rp = rng.uniform(0.0, 2.5, count)
    source_id = np.arange(1, count + 1, dtype=np.int64)
    if with_polaris:
        star = apparent.POLARIS
        moved = propagate(
            np.array([star.ra_deg]),
            np.array([star.dec_deg]),
            np.array([star.pm_ra_mas_yr]),
            np.array([star.pm_dec_mas_yr]),
            apparent.CATALOG_EPOCH_JYEAR - star.epoch_jyear,
        )
        polaris_ra, polaris_dec = vector_to_radec(moved)
        ra = np.append(ra, polaris_ra)
        dec = np.append(dec, polaris_dec)
        g_mag = np.append(g_mag, 1.95)
        pm_ra = np.append(pm_ra, star.pm_ra_mas_yr)
        pm_dec = np.append(pm_dec, star.pm_dec_mas_yr)
        parallax = np.append(parallax, star.parallax_mas)
        bp_rp = np.append(bp_rp, 0.9)
        source_id = np.append(source_id, -4_628_002_371)
    return CapCatalog.from_columns(
        source_id=source_id,
        ra_deg=ra,
        dec_deg=dec,
        g_mag=g_mag,
        pm_ra_mas_yr=pm_ra,
        pm_dec_mas_yr=pm_dec,
        parallax_mas=parallax,
        bp_rp=bp_rp,
        cap_radius_deg=cap_radius_deg,
        gaia_mag_limit=faint_limit,
    )


def project(
    rotation_cirs: FloatArray,
    vectors: FloatArray,
    scale_arcsec_px: float,
    parity: int,
    center: tuple[float, float],
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]:
    """Pixel positions of CIRS unit vectors, and a mask of those in front of the camera.

    This is the camera model of the whole survey path, written out again for the tests: the
    camera vector is `w = R u`, the gnomonic coordinates are `w_x / w_z` and `w_y / w_z`, and
    the pixel is the center plus those over the scale, with the y axis flipped for a mirrored
    optical path (`parity = -1`).
    """
    camera = vectors @ rotation_cirs.T
    in_front = camera[:, 2] > 0.05
    safe = np.where(in_front, camera[:, 2], 1.0)
    scale_rad = scale_arcsec_px / ARCSEC_PER_RAD
    x = center[0] + camera[:, 0] / safe / scale_rad
    y = center[1] + parity * camera[:, 1] / safe / scale_rad
    return x, y, in_front


@dataclass(frozen=True, slots=True, eq=False)
class SynthTruth:
    """What the renderer drew: the pointing, and where and how bright each star in the frame is."""

    rotation_tirs: FloatArray
    rotation_cirs: FloatArray  # at the middle of the exposure
    t_utc_ns: int
    exposure_s: float
    scale_arcsec_px: float
    parity: int
    center_px: tuple[float, float]
    mode: str
    width: int
    height: int
    rows: np.ndarray[Any, np.dtype[np.intp]]  # catalog rows of the stars in the frame
    x: FloatArray  # true positions at the middle of the exposure
    y: FloatArray
    flux_e: FloatArray  # electrons that reach the camera, before noise
    trail_px: FloatArray
    pole_px: tuple[float, float]
    vectors_cirs: FloatArray  # of the whole catalog
    hot_pixels: np.ndarray[Any, np.dtype[np.bool_]]  # where the renderer put hot pixels
    epoch: apparent.ObservationEpoch = field(repr=False)
    zero_point_mag: float = 0.0  # G = ZP - 2.5 log10(rate) + color_term * (BP-RP)
    color_term: float = 0.0
    sky_e_per_s_px: float = 0.0
    dark_e_per_s_px: float = 0.0
    offset_dn: float = 0.0
    temperature_c: float = 15.0


def astropy_apparent_vectors(catalog: CapCatalog, t_utc_ns: int) -> FloatArray:
    """The apparent CIRS unit vectors of the whole catalog from `astropy`, which is the truth."""
    import astropy.units as units
    import erfa
    from astropy.coordinates import CIRS, SkyCoord
    from astropy.time import Time
    from astropy.utils.exceptions import AstropyWarning

    parallax = np.maximum(catalog.parallax_mas, 1e-4)
    stars = SkyCoord(
        ra=catalog.ra_deg * units.deg,
        dec=catalog.dec_deg * units.deg,
        pm_ra_cosdec=catalog.pm_ra_mas_yr * units.mas / units.yr,
        pm_dec=catalog.pm_dec_mas_yr * units.mas / units.yr,
        distance=(1000.0 / parallax) * units.pc,
        radial_velocity=np.zeros(len(catalog)) * units.km / units.s,
        frame="icrs",
        obstime=Time("J2016.0"),
    )
    observed = Time(t_utc_ns / NS_PER_S, format="unix", scale="utc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", erfa.ErfaWarning)
        warnings.simplefilter("ignore", AstropyWarning)
        cirs = stars.apply_space_motion(new_obstime=observed).transform_to(CIRS(obstime=observed))
    return radec_to_vector(cirs.ra.deg, cirs.dec.deg)


def _draw_stars(
    image: npt.NDArray[np.float32],
    xs: FloatArray,
    ys: FloatArray,
    flux: FloatArray,
    sigma: float,
    half: np.ndarray[Any, np.dtype[np.intp]],
) -> None:
    """Add each star to `image`. `xs` and `ys` have shape `(N, K)`: K positions along the trail."""
    height, width = image.shape
    n, k = xs.shape
    root2 = np.sqrt(2.0)
    for i in range(n):
        h = int(half[i])
        cx, cy = round(float(xs[i].mean())), round(float(ys[i].mean()))
        px = np.arange(cx - h, cx + h + 1)
        py = np.arange(cy - h, cy + h + 1)
        edges_x = np.concatenate([px - 0.5, [px[-1] + 0.5]])
        edges_y = np.concatenate([py - 0.5, [py[-1] + 0.5]])
        cdf_x = erf((edges_x[None, :] - xs[i][:, None]) / (sigma * root2))
        cdf_y = erf((edges_y[None, :] - ys[i][:, None]) / (sigma * root2))
        phi_x = 0.5 * (cdf_x[:, 1:] - cdf_x[:, :-1])  # (K, S)
        phi_y = 0.5 * (cdf_y[:, 1:] - cdf_y[:, :-1])
        stamp = (phi_y.T @ phi_x) * (flux[i] / k)
        y0, y1 = max(py[0], 0), min(py[-1] + 1, height)
        x0, x1 = max(px[0], 0), min(px[-1] + 1, width)
        if y0 < y1 and x0 < x1:
            image[y0:y1, x0:x1] += stamp[y0 - py[0] : y1 - py[0], x0 - px[0] : x1 - px[0]].astype(
                np.float32
            )


def star_truth(
    catalog: CapCatalog,
    profile: Profile,
    *,
    rotation_tirs: FloatArray,
    t_utc_ns: int = NIGHT_UTC_NS,
    exposure_s: float = 30.0,
    mode: str = "bin2",
    scale_error: float = 0.0,
    parity: int = 1,
    transmission: float | TransmissionFunction = 1.0,
    exclude_rows: tuple[int, ...] = (),
    vectors_cirs: FloatArray | None = None,
    n_sub: int = 64,
    dut1_s: float = 0.0,
    zero_point_mag: float | None = None,
    color_term: float = 0.0,
) -> tuple[SynthTruth, FloatArray, FloatArray]:
    """The geometry of a frame without its pixels: the truth, and the star tracks.

    The tracks `xs` and `ys` have shape `(N, K)`: where each star sits at `K` moments during
    the exposure. The Earth turns, and the camera with it, so the camera attitude changes by a
    rotation about the pole, and far stars trail.
    """
    readout = profile.mode(mode)
    width, height = readout.width_px, readout.height_px
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    scale = profile.plate_scale_arcsec_per_px(mode) * (1.0 + scale_error)
    epoch = apparent.epoch_from_utc_ns(t_utc_ns, dut1_s)
    if vectors_cirs is None:
        vectors_cirs = apparent.apparent_vectors(
            catalog.ra_deg,
            catalog.dec_deg,
            catalog.pm_ra_mas_yr,
            catalog.pm_dec_mas_yr,
            catalog.parallax_mas,
            epoch,
            catalog_epoch_jyear=catalog.epoch_jyear,
        )
    rotation_cirs = rotation_tirs @ rot_z(-epoch.era_rad)
    offsets = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    mid_x, mid_y, front = project(rotation_cirs, vectors_cirs, scale, parity, center)
    margin = 30.0
    inside = front & (mid_x > -margin) & (mid_x < width + margin) & (mid_y > -margin)
    inside &= mid_y < height + margin
    inside[list(exclude_rows)] = False
    rows = np.flatnonzero(inside)
    xs = np.empty((rows.size, n_sub))
    ys = np.empty((rows.size, n_sub))
    for k, offset in enumerate(offsets):
        rotation_k = rotation_cirs @ rot_z(
            -apparent.EARTH_ROTATION_RATE_RAD_S * exposure_s * offset
        )
        x_k, y_k, _ = project(rotation_k, vectors_cirs[rows], scale, parity, center)
        xs[:, k], ys[:, k] = x_k, y_k
    trail = np.hypot(xs[:, -1] - xs[:, 0], ys[:, -1] - ys[:, 0]) * n_sub / (n_sub - 1)
    gain_factor = (
        np.full(rows.size, float(transmission))
        if isinstance(transmission, int | float)
        else np.asarray(transmission(xs.mean(axis=1), ys.mean(axis=1)), dtype=np.float64)
    )
    zero_point = prior_zero_point(profile) if zero_point_mag is None else zero_point_mag
    bp_rp = np.nan_to_num(catalog.bp_rp[rows], nan=0.0)
    rate = 10.0 ** (0.4 * (zero_point - catalog.g_mag[rows] + color_term * bp_rp))
    flux_e = rate * exposure_s * gain_factor
    pole_x, pole_y, pole_front = project(
        rotation_cirs, np.array([[0.0, 0.0, 1.0]]), scale, parity, center
    )
    assert pole_front[0]
    truth = SynthTruth(
        rotation_tirs=rotation_tirs,
        rotation_cirs=rotation_cirs,
        t_utc_ns=t_utc_ns,
        exposure_s=exposure_s,
        scale_arcsec_px=scale,
        parity=parity,
        center_px=center,
        mode=mode,
        width=width,
        height=height,
        rows=rows,
        x=xs.mean(axis=1),
        y=ys.mean(axis=1),
        flux_e=flux_e,
        trail_px=trail,
        pole_px=(float(pole_x[0]), float(pole_y[0])),
        vectors_cirs=vectors_cirs,
        hot_pixels=np.zeros((height, width), dtype=np.bool_),
        epoch=epoch,
        zero_point_mag=zero_point,
        color_term=color_term,
    )
    return truth, xs, ys


def render_frame(
    catalog: CapCatalog,
    profile: Profile,
    *,
    rotation_tirs: FloatArray,
    t_utc_ns: int = NIGHT_UTC_NS,
    exposure_s: float = 30.0,
    mode: str = "bin2",
    gain: int = 120,
    psf_sigma_px: float = 0.45,
    sky_e_per_s_px: float = 2.7,
    sky_gradient: float = 0.0,
    scale_error: float = 0.0,
    parity: int = 1,
    transmission: float | TransmissionFunction = 1.0,
    halo_fraction: float = 0.02,
    halo_sigma_px: float = 10.0,
    n_hot_pixels: int = 0,
    hot_e_per_s: float = 40.0,
    offset_dn: float = 40.0,
    seed: int = 0,
    exclude_rows: tuple[int, ...] = (),
    vectors_cirs: FloatArray | None = None,
    stream_id: int = 1,
    seq: int = 0,
    n_sub: int = 64,
    dut1_s: float = 0.0,
    zero_point_mag: float | None = None,
    color_term: float = 0.0,
    star_scatter: float = 0.0,
    dark_e_per_s_px: float = 0.0,
    temperature_c: float = 15.0,
) -> tuple[Frame, SynthTruth]:
    """Render one survey frame and return it with the truth.

    The true plate scale is the profile's times `1 + scale_error`. `transmission` is a number
    or a function of the pixel position that scales the light of every star, which simulates
    clouds. `sky_gradient` varies the sky linearly across the columns, by that fraction of its
    level on each side of the middle. A star with more than 30 noise levels in the halo gets a
    halo of `halo_fraction` of its flux and width `halo_sigma_px`. `vectors_cirs` replaces the
    apparent places (use `astropy_apparent_vectors` for an independent truth).
    """
    rng = np.random.default_rng(seed)
    truth, xs, ys = star_truth(
        catalog,
        profile,
        rotation_tirs=rotation_tirs,
        t_utc_ns=t_utc_ns,
        exposure_s=exposure_s,
        mode=mode,
        scale_error=scale_error,
        parity=parity,
        transmission=transmission,
        exclude_rows=exclude_rows,
        vectors_cirs=vectors_cirs,
        n_sub=n_sub,
        dut1_s=dut1_s,
        zero_point_mag=zero_point_mag,
        color_term=color_term,
    )
    if star_scatter > 0.0:  # scintillation and other flux noise that the photometry cannot model
        factor = np.exp(rng.normal(0.0, star_scatter, truth.rows.size))
        truth = replace(truth, flux_e=truth.flux_e * factor)
    readout = profile.mode(mode)
    width, height = truth.width, truth.height
    e_per_adu = profile.e_per_adu(mode, gain)
    read_noise = profile.read_noise_e(mode, gain)
    saturation = profile.saturation(mode, gain)
    flux_e, trail = truth.flux_e, truth.trail_px

    electrons = np.zeros((height, width), dtype=np.float32)
    sigma = psf_sigma_px
    # A star that will saturate gets a halo, which spreads its wings over many pixels.
    sky_e = (sky_e_per_s_px + dark_e_per_s_px) * exposure_s
    noise_e = np.sqrt(sky_e + read_noise**2)
    halo = flux_e * halo_fraction > 30.0 * noise_e * np.sqrt(2.0 * np.pi) * halo_sigma_px
    core_flux = np.where(halo, flux_e * (1.0 - halo_fraction), flux_e)
    half = np.ceil(5.0 * sigma + trail / 2.0 + 2.0).astype(np.intp)
    _draw_stars(electrons, xs, ys, core_flux, sigma, half)
    if halo.any():
        halo_half = np.full(int(halo.sum()), int(np.ceil(5.0 * halo_sigma_px + 2.0)), dtype=np.intp)
        _draw_stars(
            electrons,
            xs[halo],
            ys[halo],
            flux_e[halo] * halo_fraction,
            halo_sigma_px,
            halo_half,
        )
    ramp = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    electrons += np.float32(sky_e) * (1.0 + np.float32(sky_gradient) * ramp)[None, :]
    hot_mask = np.zeros((height, width), dtype=np.bool_)
    if n_hot_pixels:
        hot_y = rng.integers(0, height, n_hot_pixels)
        hot_x = rng.integers(0, width, n_hot_pixels)
        electrons[hot_y, hot_x] += np.float32(hot_e_per_s * exposure_s)
        hot_mask[hot_y, hot_x] = True

    native = np.empty((height, width), dtype=np.float32)
    for start in range(0, height, 256):
        block = electrons[start : start + 256]
        counts = rng.poisson(block) + rng.normal(0.0, read_noise, block.shape)
        native[start : start + 256] = np.rint(counts / e_per_adu + offset_dn)
    np.clip(native, 0.0, saturation.native_dn, out=native)
    shift = 16 - readout.adc_bits
    data = (native.astype(np.uint32) << shift).astype(np.uint16)

    frame = Frame(
        data=data,
        stream_id=stream_id,
        seq=seq,
        t_arrival_ns=t_utc_ns + round(exposure_s * NS_PER_S / 2),
        t_utc_ns=t_utc_ns,
        t_err_ns=0,
        t_quality=TimeQuality.EXACT,
        dropped_before=0,
        exposure_us=round(exposure_s * 1e6),
        gain=gain,
        mode=mode,
        roi=Roi(0, 0, width, height),
        adc_bits=readout.adc_bits,
        temperature_c=temperature_c,
        flags=FrameFlag.SIMULATED,
    )
    return frame, replace(
        truth,
        hot_pixels=hot_mask,
        sky_e_per_s_px=sky_e_per_s_px,
        dark_e_per_s_px=dark_e_per_s_px,
        offset_dn=offset_dn,
        temperature_c=temperature_c,
    )


def truth_solve_result(
    truth: SynthTruth,
    catalog: CapCatalog,
    *,
    solver: str = "synthetic",
    center_shift_px: float = 0.0,
) -> SolveResult:
    """The solution that a good solver finds: a linear TAN fit to the stars in the frame.

    A real solver works in the catalog frame (ICRS at the catalog epoch, with no aberration),
    so this fits a tangent-plane map from pixel offsets to ICRS gnomonic coordinates around the
    ICRS direction of the image center, with a translation. `center_shift_px` moves the
    reference direction along x, for a solver that is a little off.
    """
    boresight_cirs = truth.rotation_cirs.T @ np.array([0.0, 0.0, 1.0])
    boresight = apparent.astrometric_from_apparent(boresight_cirs, truth.epoch)
    ra0, dec0 = vector_to_radec(boresight)
    east, north, outward = tangent_basis(float(ra0), float(dec0))
    inside = (
        (truth.x > 0) & (truth.x < truth.width - 1) & (truth.y > 0) & (truth.y < truth.height - 1)
    )
    u = catalog.vectors[truth.rows[inside]]
    xi = (u @ east) / (u @ outward)
    eta = (u @ north) / (u @ outward)
    dx = truth.x[inside] - truth.center_px[0]
    dy = truth.y[inside] - truth.center_px[1]
    design = np.column_stack([dx, dy, np.ones_like(dx)])
    row_xi, *_ = np.linalg.lstsq(design, xi, rcond=None)
    row_eta, *_ = np.linalg.lstsq(design, eta, rcond=None)
    cd = np.array([[row_xi[0], row_xi[1]], [row_eta[0], row_eta[1]]])
    shift = np.array([row_xi[2], row_eta[2]]) + cd @ np.array([center_shift_px, 0.0])
    reference = boresight + shift[0] * east + shift[1] * north
    ra, dec = vector_to_radec(reference / np.linalg.norm(reference))
    cd_deg = cd * (180.0 / np.pi)
    residual = np.hypot(design @ row_xi - xi, design @ row_eta - eta)
    return SolveResult(
        solved=True,
        solver=solver,
        elapsed_s=0.25,
        center_ra_deg=float(ra),
        center_dec_deg=float(dec),
        scale_arcsec_px=float(np.sqrt(abs(np.linalg.det(cd))) * ARCSEC_PER_RAD),
        cd_matrix=(
            float(cd_deg[0, 0]),
            float(cd_deg[0, 1]),
            float(cd_deg[1, 0]),
            float(cd_deg[1, 1]),
        ),
        n_matched=int(inside.sum()),
        rms_arcsec=float(np.sqrt(np.mean(residual**2)) * ARCSEC_PER_RAD),
    )


class QueueSolver:
    """A `PlateSolver` that returns scripted results in order, then reports failure."""

    def __init__(self, results: list[SolveResult] | None = None, name: str = "synthetic") -> None:
        self._results = list(results or [])
        self._name = name
        self.requests: list[SolveRequest] = []

    @property
    def name(self) -> str:
        return self._name

    def add(self, result: SolveResult) -> None:
        self._results.append(result)

    def solve(self, request: SolveRequest) -> SolveResult:
        self.requests.append(request)
        if self._results:
            return self._results.pop(0)
        return SolveResult(solved=False, solver=self._name, elapsed_s=0.0)
