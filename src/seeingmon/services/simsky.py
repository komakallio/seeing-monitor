"""A simulated sky for development, the dev launcher, and the end-to-end tests.

The system needs three things that a simulated run must agree on: the camera (the `sim` driver in
`acquire`), the cap catalog (which the survey analysis in `core` matches stars against), and a first
pointing solution (which a real installation gets from a plate solver, and which a development
machine without the solver programs does not have). This module makes all three from one seed:

- `write_small_profile` copies the reference profile with a small sensor. A 640 by 480 bin2 frame
  (1280 by 960 in bin1) keeps the simulator and the survey analysis fast, and the optics stay, so
  the plate scale is the real one.
- `sim_field` and `sim_catalog` make the star field that the simulator renders for a seed, and the
  cap catalog of the same stars, so that every simulated star is a catalog star. Polaris sits where
  the survey code predicts the real one, so the star that the scheduler follows is the star that
  the simulator draws.
- `seed_solution` computes the pointing solution that the simulated camera has at a time. It fits
  the rotation that carries the apparent places of the catalog stars onto the pixel positions that
  the simulator gives them (a Kabsch fit with the known pairs), for both parities of the image.
  `core` loads the solution as the first state of its pointing tracker (the setting
  `[services.core] seed_solution_file`), and the survey analysis tracks the field from there.

The simulator turns the sky about its own pole, which sits 0.618 degrees from its Polaris, and the
survey code turns it about the true pole of date. The two agree at the time of the seed solution and
drift apart by about 0.4 pixel a minute (27 pixels an hour in bin2). The survey step renews the
solution every few minutes, so a simulated run follows the star, but a solution that is an hour old
points at empty sky.

The helpers need the simulator and the survey path, so import this module only where you use them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seeingmon import paths
from seeingmon.drivers.sim.params import SimParams
from seeingmon.drivers.sim.stars import (
    Pointing,
    SkyProjector,
    StarField,
    make_polar_field,
    polaris_field,
)
from seeingmon.profile import Profile
from seeingmon.survey import apparent
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.catalog_build import propagate
from seeingmon.survey.geometry import ARCSEC_PER_RAD, vector_to_radec
from seeingmon.survey.pointing import PointingSolution
from seeingmon.survey.wcs_fit import CameraAttitude, pixel_center

SMALL_BIN2 = (640, 480)
MIN_SEED_STARS = 12


def write_small_profile(
    directory: Path, name: str = "sim-small", size: tuple[int, int] = SMALL_BIN2
) -> Path:
    """Write the reference profile with a smaller sensor, and return the path.

    `size` is the bin2 width and height. Bin1 has twice that. The id of the profile is `name`, and
    so is the stem of the file, as the loader requires.
    """
    width, height = size
    text = (paths.profiles_dir() / "asi294mm-gs250.toml").read_text(encoding="utf-8")
    text = text.replace('id = "asi294mm-gs250"', f'id = "{name}"', 1)
    for old, new in (
        ("width_px = 8288", f"width_px = {2 * width}"),
        ("height_px = 5644", f"height_px = {2 * height}"),
        ("width_px = 4144", f"width_px = {width}"),
        ("height_px = 2822", f"height_px = {height}"),
    ):
        if old not in text:
            raise ValueError(f"the reference profile has no line {old!r}")
        text = text.replace(old, new, 1)
    path = Path(directory) / f"{name}.toml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def polaris_rows(field: StarField) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """The rows of Polaris and its companion in a field of `make_polar_field`.

    The function appends the two stars after the random stars, so they are the last rows. The
    brightest row of the field is a random star of magnitude 0, and not Polaris.
    """
    count = len(polaris_field().mag)
    return np.arange(len(field.mag) - count, len(field.mag), dtype=np.intp)


def sim_field(seed: int = 1, *, polaris_mag: float | None = None) -> StarField:
    """The star field that the simulator renders for `seed`, with Polaris where the survey puts it.

    The synthetic field of the simulator has a Polaris at an approximate place. The pointing
    tracker predicts the real Polaris from its astrometry, so the field gets the real place of
    Polaris at the catalog epoch, and Polaris B moves by the same amount. Every simulated star is
    then a catalog star, and the star that the scheduler follows is the star that the simulator
    draws. `polaris_mag` makes Polaris and its companion fainter by the difference between
    this magnitude and the magnitude of Polaris, for a run whose exposure would saturate the
    real star.
    """
    field = make_polar_field(seed)
    star = apparent.POLARIS
    moved = propagate(
        np.array([star.ra_deg]),
        np.array([star.dec_deg]),
        np.array([star.pm_ra_mas_yr]),
        np.array([star.pm_dec_mas_yr]),
        apparent.CATALOG_EPOCH_JYEAR - star.epoch_jyear,
    )
    ra_new, dec_new = vector_to_radec(moved)
    rows = polaris_rows(field)
    polaris = int(rows[0])
    delta_ra = float(ra_new[0]) - float(field.ra_deg[polaris])
    delta_dec = float(dec_new[0]) - float(field.dec_deg[polaris])
    ra = field.ra_deg.copy()
    dec = field.dec_deg.copy()
    mag = field.mag.copy()
    ra[rows] += delta_ra  # Polaris B keeps its place relative to Polaris
    dec[rows] += delta_dec
    if polaris_mag is not None:
        mag[rows] += polaris_mag - float(field.mag[polaris])
    return StarField(ra, dec, mag)


def sim_catalog(seed: int = 1, *, polaris_mag: float | None = None) -> tuple[CapCatalog, StarField]:
    """The cap catalog of the star field that the simulator renders for `seed`, and the field."""
    field = sim_field(seed, polaris_mag=polaris_mag)
    star = apparent.POLARIS
    polaris = int(polaris_rows(field)[0])
    pm_ra = np.zeros(len(field.mag))
    pm_dec = np.zeros(len(field.mag))
    parallax = np.zeros(len(field.mag))
    pm_ra[polaris], pm_dec[polaris] = star.pm_ra_mas_yr, star.pm_dec_mas_yr
    parallax[polaris] = star.parallax_mas
    catalog = CapCatalog.from_columns(
        source_id=np.arange(1, len(field.mag) + 1, dtype=np.int64),
        ra_deg=field.ra_deg,
        dec_deg=field.dec_deg,
        g_mag=field.mag,
        pm_ra_mas_yr=pm_ra,
        pm_dec_mas_yr=pm_dec,
        parallax_mas=parallax,
        cap_radius_deg=15.0,
        gaia_mag_limit=float(field.mag.max()),
        notes="simulated sky",
    )
    return catalog, field


def _kabsch(
    vectors: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    rays: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """The proper rotation `R` that minimizes `|rays - vectors @ R.T|` over the pairs."""
    h = vectors.T @ rays
    u, _, vt = np.linalg.svd(h)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign if sign != 0 else 1.0])
    return np.asarray(vt.T @ correction @ u.T, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SeedFit:
    """The seed solution and how well it fits the pairs that made it."""

    solution: PointingSolution
    n_stars: int
    rms_px: float
    parity: int


def seed_solution(
    profile: Profile,
    field: StarField,
    pointing: Pointing,
    t_utc_ns: int,
    *,
    mode: str | None = None,
    max_stars: int = 400,
) -> SeedFit:
    """The pointing solution of the simulated camera at `t_utc_ns`, in the survey readout mode.

    The brightest `max_stars` stars that fall inside the frame give the pairs. Raises `ValueError`
    when fewer than `MIN_SEED_STARS` stars fall inside, which means that the pointing and the
    profile disagree.
    """
    mode = mode or profile.survey_mode.mode
    readout = profile.mode(mode)
    params = SimParams.modes_from_profile(profile)[mode]
    projector = SkyProjector(field, pointing)
    x, y = projector.project(t_utc_ns, params.pixel_rad, params.width, params.height)
    stars = projector.stars
    inside = (x > 2) & (x < params.width - 3) & (y > 2) & (y < params.height - 3)
    order = np.argsort(stars.mag)
    chosen = [int(i) for i in order if inside[i]][:max_stars]
    if len(chosen) < MIN_SEED_STARS:
        raise ValueError(f"only {len(chosen)} simulated stars fall inside the frame")
    pick = np.asarray(chosen)
    epoch = apparent.epoch_from_utc_ns(t_utc_ns, 0.0)
    zeros = np.zeros(len(pick))
    vectors = apparent.apparent_vectors(
        stars.ra_deg[pick],
        stars.dec_deg[pick],
        zeros,
        zeros,
        zeros,
        epoch,
        catalog_epoch_jyear=apparent.CATALOG_EPOCH_JYEAR,
    )
    scale_rad = profile.plate_scale_arcsec_per_px(readout) / ARCSEC_PER_RAD
    center = pixel_center(readout.width_px, readout.height_px)
    best: SeedFit | None = None
    for parity in (1, -1):
        xi = (x[pick] - center[0]) * scale_rad
        eta = parity * (y[pick] - center[1]) * scale_rad
        rays = np.stack([xi, eta, np.ones_like(xi)], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        rotation = _kabsch(vectors, rays)
        attitude = CameraAttitude(rotation, scale_rad, parity, center)
        px, py, front = attitude.project(vectors)
        if not front.all():
            continue
        rms = float(np.sqrt(np.mean((px - x[pick]) ** 2 + (py - y[pick]) ** 2)))
        if best is None or rms < best.rms_px:
            solution = PointingSolution.from_attitude(
                attitude,
                epoch,
                mode=mode,
                width_px=readout.width_px,
                height_px=readout.height_px,
                n_matched=len(pick),
                rms_arcsec=rms * scale_rad * ARCSEC_PER_RAD,
                solver="sim-seed",
            )
            best = SeedFit(solution, len(pick), rms, parity)
    if best is None:
        raise ValueError("no parity puts the simulated stars in front of the camera")
    return best


def write_seed(path: Path, solution: PointingSolution) -> None:
    """Write a solution as JSON, for `[services.core] seed_solution_file`."""
    Path(path).write_text(json.dumps(solution.to_dict(), indent=2), encoding="utf-8", newline="\n")


def read_seed(path: Path | str) -> PointingSolution:
    """Read a solution that `write_seed` wrote."""
    return PointingSolution.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
