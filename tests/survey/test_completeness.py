"""The completeness model of the detector's searches, against stars that the detector must find.

`seeingmon.survey.completeness.SearchModel` predicts the chance that a search of
`seeingmon.survey.detect` finds a star of a given flux. The tests inject a grid of stars at random
positions within a 2 x 2 block (the period of the binned search) into Gaussian noise, run
`detect_stars`, and compare the share that it finds with the model. The image of a star is
rendered here on a grid of 5 x 5 points in each pixel, independently of the model's integral over
the pixel. The last tests draw the stars with the simulator's image of a long exposure, whose Airy
rings hold light that a Gaussian lacks.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from seeingmon.drivers.sim.optics import MixturePsf, PsfConfig
from seeingmon.drivers.sim.params import SimParams
from seeingmon.drivers.sim.turbulence import g_tilt_variance_rad2
from seeingmon.survey.centroid import FWHM_PER_SIGMA
from seeingmon.survey.completeness import (
    ANGLE_NODES_RAD,
    SearchModel,
    fold_angle,
    model_fwhm,
    star_completeness,
    star_limit_e,
)
from seeingmon.survey.detect import DetectOptions, NoiseMap, SearchSpec, detect_stars
from seeingmon.survey.photometry import PhotometryOptions
from seeingmon.survey.pipeline import light_in_core
from seeingmon.survey.trail import TrailModel
from tests.survey import synth

FloatArray = npt.NDArray[np.float64]

NOISE_E = 66.0  # the pixel noise of a frame of 1 s at the target of the twilight exposure
SPACING = 24  # pixels between the injected stars
STARS_PER_SIDE = 12
SEEDS = (0, 1, 2)  # 3 x 144 stars a case: the binomial error of the share is 0.024 at most
FULL = SearchSpec(5.0, 2, 1, relative=False)
FULL_RELATIVE = SearchSpec(5.0, 2, 1, relative=True)
BINNED = SearchSpec(5.0, 2, 2, relative=False)
# The binned search needs a trail model to run. A pole far away with no rotation gives no trail, so
# the fit of the detector uses none, and the stars keep the trail of the injection.
NO_ROTATION = TrailModel(-1e7, -1e7, 0.0)
_SUPERSAMPLE = 5


def render_star(
    image: FloatArray,
    x: float,
    y: float,
    flux: float,
    fwhm_px: float,
    trail_px: float,
    trail_angle_rad: float = 0.0,
) -> None:
    """Add a Gaussian star smeared along its trail, sampled at 5 x 5 points in each pixel."""
    sigma = fwhm_px / FWHM_PER_SIGMA
    reach = math.ceil(4.0 * sigma + trail_px / 2.0) + 1
    x0, y0 = round(x) - reach, round(y) - reach
    offsets = (np.arange(_SUPERSAMPLE) + 0.5) / _SUPERSAMPLE - 0.5
    columns = x0 + np.arange(2 * reach + 1)[:, None] + offsets  # (pixels, samples)
    rows = y0 + np.arange(2 * reach + 1)[:, None] + offsets
    points = max(1, math.ceil(trail_px / 0.1) + 1)
    along = np.linspace(-trail_px / 2.0, trail_px / 2.0, points) if points > 1 else np.zeros(1)
    stamp = np.zeros((rows.shape[0], columns.shape[0]))
    for shift in along:
        center_x = x + shift * math.cos(trail_angle_rad)
        center_y = y + shift * math.sin(trail_angle_rad)
        profile_y = np.exp(-0.5 * ((rows - center_y) / sigma) ** 2).mean(axis=1)
        profile_x = np.exp(-0.5 * ((columns - center_x) / sigma) ** 2).mean(axis=1)
        stamp += np.outer(profile_y, profile_x)
    image[y0 : y0 + stamp.shape[0], x0 : x0 + stamp.shape[1]] += flux * stamp / stamp.sum()


def sim_stamps() -> tuple[MixturePsf, FloatArray, FloatArray]:
    """The simulator's image of a 30 s bin2 star: Airy rings, a seeing halo, and the wander."""
    params = SimParams.from_profile(synth.cropped_profile(1200, 900), "bin2")
    psf = MixturePsf(params, PsfConfig(mode="gaussian"))
    r0 = 0.10  # the simulator's default
    wander = math.sqrt(g_tilt_variance_rad2(params.aperture_m, r0, 20.0)) / params.pixel_rad
    weights, sigmas = psf.components(r0, wander)
    return psf, weights, sigmas


def injected_frame(
    flux_sigma: float,
    seed: int,
    *,
    fwhm_px: float = 0.73,
    trail_px: float = 0.0,
    trail_angle_rad: float = 0.0,
    noisy_strip: bool = False,
    sim_psf: bool = False,
    spacing: int = SPACING,
    spread: float = 1.0,
) -> tuple[FloatArray, list[tuple[float, float]]]:
    """A frame of 144 stars of `flux_sigma` times the pixel noise, and their positions.

    The sky has `NOISE_E` electrons of noise (one electron a count), and each star adds its own
    photon noise. Each star lies anywhere within a 2 x 2 block, so the stars cover every position
    relative to the blocks of the binned search. `noisy_strip` adds a strip with twice the noise far
    from the stars, so that SEP takes the noise map and thresholds the SNR of each pixel.
    `sim_psf` draws the stars with the simulator's image instead of a Gaussian. `spacing` is the
    distance between the stars, in pixels, and it must be even. With a `spread` above 1, the fluxes
    spread evenly in their logarithm from `flux_sigma` up to `spread` times as much.
    """
    rng = np.random.default_rng(seed)
    size = spacing * STARS_PER_SIDE
    width = size + (192 if noisy_strip else 0)
    stars = np.zeros((size, width))
    truth = []
    psf, weights, sigmas = sim_stamps() if sim_psf else (None, None, None)
    for row in range(STARS_PER_SIDE):
        for column in range(STARS_PER_SIDE):
            x = spacing * column + spacing / 2 + rng.uniform(-1.0, 1.0)
            y = spacing * row + spacing / 2 + rng.uniform(-1.0, 1.0)
            flux = flux_sigma * NOISE_E * spread ** rng.uniform(0.0, 1.0)
            if psf is None:
                render_star(stars, x, y, flux, fwhm_px, trail_px, trail_angle_rad)
            else:
                assert weights is not None
                assert sigmas is not None
                ix, iy = round(x), round(y)
                stamp = psf.stamps(
                    np.array([x - ix]), np.array([y - iy]), weights, sigmas, size_px=15
                )[0]
                stars[iy - 7 : iy + 8, ix - 7 : ix + 8] += flux * stamp
            truth.append((x, y))
    noise = np.full(stars.shape, NOISE_E)
    if noisy_strip:
        noise[:, size + 96 :] *= 2.0
    frame = 1000.0 + stars + rng.normal(size=stars.shape) * np.sqrt(noise**2 + stars)
    return frame, truth


def found_share(
    flux_sigma: float,
    fwhm_px: float = 0.73,
    *,
    coarse_bin: int = 1,
    trail_px: float = 0.0,
    trail_angle_rad: float = 0.0,
    noisy_strip: bool = False,
    sim_psf: bool = False,
) -> float:
    """The share of the injected stars that the detector finds, over the three seeds."""
    shares = []
    for seed in SEEDS:
        frame, truth = injected_frame(
            flux_sigma,
            seed,
            fwhm_px=fwhm_px,
            trail_px=trail_px,
            trail_angle_rad=trail_angle_rad,
            noisy_strip=noisy_strip,
            sim_psf=sim_psf,
        )
        found = detect_stars(
            frame,
            saturation_dn=1e9,
            options=DetectOptions(coarse_bin=coarse_bin),
            e_per_adu=1.0,
            trail=NO_ROTATION if coarse_bin > 1 else None,
        )
        assert found.search is not None
        assert found.search.factor == coarse_bin
        assert found.search.relative is noisy_strip
        hits = 0
        for x, y in truth:
            if len(found) and float(np.min(np.hypot(found.x - x, found.y - y))) < 2.0:
                hits += 1
        shares.append(hits / len(truth))
    return float(np.mean(shares))


def model_share(
    search: SearchSpec,
    flux_sigma: float,
    fwhm_px: float,
    trail_px: float = 0.0,
    trail_angle_rad: float = 0.0,
) -> float:
    model = SearchModel.build(
        search, fwhm_px=fwhm_px, trail_px=trail_px, trail_angle_rad=trail_angle_rad
    )
    return float(model.completeness(flux_sigma * NOISE_E, NOISE_E))


@pytest.mark.parametrize(
    ("search", "fwhm_px", "trail_px", "angle_deg", "fluxes"),
    [
        (FULL, 0.73, 0.0, 0.0, (30.0, 36.0, 45.0)),
        (FULL, 1.5, 0.0, 0.0, (40.0, 46.0, 60.0)),
        (FULL, 2.5, 0.0, 0.0, (55.0, 62.0, 80.0)),
        (FULL, 0.75, 3.1, 30.0, (45.0, 52.0, 65.0)),
        (BINNED, 0.73, 0.0, 0.0, (40.0, 90.0, 200.0, 800.0)),
        (BINNED, 1.5, 0.0, 0.0, (40.0, 60.0, 130.0)),
        (BINNED, 0.75, 1.8, 0.0, (60.0, 130.0)),
        (BINNED, 0.75, 1.8, 45.0, (60.0, 130.0, 200.0)),
        (BINNED, 0.75, 3.1, 30.0, (40.0, 70.0, 110.0)),
    ],
)
def test_the_model_gives_the_share_of_stars_that_the_detector_finds(
    search: SearchSpec,
    fwhm_px: float,
    trail_px: float,
    angle_deg: float,
    fluxes: tuple[float, ...],
) -> None:
    """The model takes only the second-highest pixel of the searched frame, and the detector also
    accepts any other neighbor above the threshold, so the model may understate the share on the
    steep part of the curve: by 0.11 at most in the runs, which the bound of 0.15 covers. It must
    not overstate it, so a frame never expects a star that the detector cannot find. The runs
    overstate by 0.04 at most, about the binomial error of 432 stars (0.024 at most), and the
    bound is 0.05. The stars lie anywhere in a 2 x 2 block, so a trail along a diagonal meets the
    blocks as it does in a frame.
    """
    angle = math.radians(angle_deg)
    for flux in fluxes:
        model = model_share(search, flux, fwhm_px, trail_px, angle)
        measured = found_share(
            flux, fwhm_px, coarse_bin=search.factor, trail_px=trail_px, trail_angle_rad=angle
        )
        assert model - 0.05 <= measured <= model + 0.15, (flux, model, measured)


def test_a_varied_noise_map_makes_the_full_search_deeper_as_the_model_says() -> None:
    """With a noise map that varies by more than 30%, SEP thresholds the matched SNR at 5, which
    is 2.67 times lower than 5 times the pixel noise on the smoothed image."""
    for flux in (12.0, 16.0, 22.0):
        model = model_share(FULL_RELATIVE, flux, 0.73)
        measured = found_share(flux, 0.73, noisy_strip=True)
        assert model - 0.05 <= measured <= model + 0.15, (flux, model, measured)
    uniform = SearchModel.build(FULL, fwhm_px=0.73).limit_e(0.9, NOISE_E)
    varied = SearchModel.build(FULL_RELATIVE, fwhm_px=0.73).limit_e(0.9, NOISE_E)
    assert uniform / varied == pytest.approx(2.4, abs=0.2)  # the runs give 39.7 / 16.6


def test_the_binned_search_loses_sharp_stars_that_a_trail_brings_back() -> None:
    """The method's own loss: a sharp star in the middle of a block puts almost no light into a
    second block, so without a trail the binned search finds 0.76 of the sharp stars at 200 times
    the pixel noise, where the full search finds every star from 45 times. The trail of a 30 s
    frame near the pole, about 1.8 px, spreads the star across blocks. The figures are those of the
    text of `seeingmon.survey.completeness`, and the tolerances hold the grid of the model.
    """
    sharp = SearchModel.build(BINNED, fwhm_px=0.73)
    full = SearchModel.build(FULL, fwhm_px=0.73)
    flux = np.array([45.0, 200.0]) * NOISE_E
    assert full.completeness(flux, NOISE_E) == pytest.approx([0.99, 1.0], abs=0.01)
    assert sharp.completeness(flux, NOISE_E) == pytest.approx([0.40, 0.75], abs=0.02)
    trailed = SearchModel.build(BINNED, fwhm_px=0.75, trail_px=1.8)
    assert float(trailed.completeness(130.0 * NOISE_E, NOISE_E)) > 0.9
    diagonal = SearchModel.build(BINNED, fwhm_px=0.75, trail_px=1.8, trail_angle_rad=math.pi / 4)
    assert float(diagonal.completeness(130.0 * NOISE_E, NOISE_E)) > 0.9
    # A wide image fills the blocks, and binning loses little.
    wide_binned = SearchModel.build(BINNED, fwhm_px=2.5).limit_e(0.9, NOISE_E)
    wide_full = SearchModel.build(FULL, fwhm_px=2.5).limit_e(0.9, NOISE_E)
    assert wide_binned < wide_full  # 66 against 71 times the pixel noise


def test_the_limit_is_the_flux_of_the_asked_completeness() -> None:
    model = SearchModel.build(FULL, fwhm_px=1.0)
    for chance in (0.5, 0.9, 0.99):
        limit = model.limit_e(chance, NOISE_E)
        assert float(model.completeness(limit, NOISE_E)) == pytest.approx(chance, abs=1e-6)
    assert model.limit_e(0.9, 2.0 * NOISE_E) == pytest.approx(
        2.0 * model.limit_e(0.9, NOISE_E), rel=0.05
    )  # the star's own noise keeps it from exactly double
    assert math.isinf(model.limit_e(0.9, 0.0))
    with pytest.raises(ValueError, match="finite"):
        SearchModel.build(FULL, fwhm_px=math.nan)
    assert np.all(model.completeness([0.0, -5.0], NOISE_E) == 0.0)
    assert float(model.completeness(1e6, NOISE_E)) == pytest.approx(1.0)
    # One noise for each star: a star with twice the noise needs about twice the flux.
    twice = model.completeness([30.0 * NOISE_E, 60.0 * NOISE_E], [NOISE_E, 2.0 * NOISE_E])
    assert twice[1] == pytest.approx(twice[0], abs=0.03)
    assert float(model.completeness(30.0 * NOISE_E, -1.0)) == 0.0


def test_a_trail_angle_folds_into_the_first_45_degrees() -> None:
    """The pixels and the blocks look the same after a turn of 90 degrees or a mirror, so a trail
    at 100 degrees meets them as one at 10 degrees does, and one at 60 degrees as one at 30."""
    angles = np.radians([0.0, 10.0, 60.0, 100.0, -30.0, 135.0, 180.0])
    np.testing.assert_allclose(
        np.degrees(fold_angle(angles)), [0.0, 10.0, 30.0, 10.0, 30.0, 45.0, 0.0], atol=1e-9
    )
    folded = SearchModel.build(BINNED, fwhm_px=0.75, trail_px=2.5, trail_angle_rad=math.radians(30))
    turned = SearchModel.build(BINNED, fwhm_px=0.75, trail_px=2.5, trail_angle_rad=math.radians(60))
    flux = np.array([60.0, 90.0, 150.0]) * NOISE_E
    np.testing.assert_allclose(
        folded.completeness(flux, NOISE_E), turned.completeness(flux, NOISE_E), atol=1e-9
    )
    assert model_fwhm(0.73) == pytest.approx(0.73, rel=0.01)  # the nearest step of 2%
    assert model_fwhm(0.731) == model_fwhm(0.733)


def test_each_star_takes_the_least_complete_corner_of_its_cell() -> None:
    """The model of each star comes from a grid of trail lengths and directions, and it takes the
    highest limit of the four corners of its cell. Against the model of the star's own trail, over
    lengths of 0 to 7 px and every direction, the limit of the full search errs high by 4% in the
    median and by at most 25%. The binned search changes fastest at short trails, where the grid
    errs high by up to 45%. Where a trail longer than 5.8 px meets the blocks at about 28 degrees,
    the binned limit has a bump between the corners, and the grid states it up to 5% low, so the
    bound is 0.94.
    """
    rng = np.random.default_rng(3)
    for search, low, high in ((BINNED, 0.94, 1.5), (FULL, 0.995, 1.3)):
        ratios = []
        for fwhm in (0.73, 1.0, 1.5, 2.5):
            length = rng.uniform(0.0, 7.0, 40)
            angle = rng.uniform(-math.pi / 2.0, math.pi / 2.0, 40)
            grid = star_limit_e(
                search, 0.9, NOISE_E, fwhm_px=fwhm, trail_px=length, trail_angle_rad=angle
            )
            exact = [
                SearchModel.build(
                    search,
                    fwhm_px=model_fwhm(fwhm),
                    trail_px=float(trail_px),
                    trail_angle_rad=float(a),
                ).limit_e(0.9, NOISE_E)
                for trail_px, a in zip(length, angle, strict=True)
            ]
            ratios.append(grid / np.array(exact))
        ratio = np.concatenate(ratios)
        assert low <= float(ratio.min())
        assert float(ratio.max()) <= high
        assert 1.0 < float(np.median(ratio)) < 1.1
    # The completeness of a star at its limit is the chance asked, and it falls with the noise.
    length = np.array([0.0, 0.6, 2.2, 6.0])
    angle = np.array([0.0, 0.3, 1.0, 2.0])
    limits = star_limit_e(
        BINNED, 0.9, NOISE_E, fwhm_px=0.75, trail_px=length, trail_angle_rad=angle
    )
    found = star_completeness(
        BINNED, limits, NOISE_E, fwhm_px=0.75, trail_px=length, trail_angle_rad=angle
    )
    np.testing.assert_allclose(found, 0.9, atol=1e-4)
    noisier = star_completeness(
        BINNED, limits, 1.5 * NOISE_E, fwhm_px=0.75, trail_px=length, trail_angle_rad=angle
    )
    assert np.all(noisier < found)
    assert ANGLE_NODES_RAD[-1] == pytest.approx(math.pi / 4.0)


def test_the_stars_near_the_pole_drop_out_of_the_binned_search() -> None:
    """A trail model with the pole in the frame, as in a 30 s frame of the full sensor: the trails
    grow from 0 at the pole to 6.5 px at the far corner, in every direction. One model for the
    median trail (3.3 px) along the mean axis of the trails, which lands near a pixel axis, the
    most favorable direction, finds a star of 51 times the pixel noise with a chance of 0.9. With
    its own trail, a star of the field finds it with a chance of 0.79 on average, and only 28% of
    the stars reach 0.9. The stars within 120 px of the pole, whose trails are shorter than 0.3 px,
    stay below 0.5. The model of each star sees this, and its grid errs low, so the cloud fraction
    leaves those stars out.
    """
    trail = TrailModel.for_exposure(1800.0, 900.0, 30.0)
    grid_x, grid_y = np.meshgrid(np.linspace(20.0, 4120.0, 41), np.linspace(20.0, 2800.0, 28))
    x, y = grid_x.ravel(), grid_y.ravel()
    length, angle = trail.length(x, y), trail.angle(x, y)
    double = 2.0 * angle  # a trail has an axis, not a direction
    axis = 0.5 * math.atan2(float(np.mean(np.sin(double))), float(np.mean(np.cos(double))))
    one = SearchModel.build(
        BINNED, fwhm_px=0.75, trail_px=float(np.median(length)), trail_angle_rad=axis
    )
    flux = one.limit_e(0.9, NOISE_E)
    assert flux / NOISE_E == pytest.approx(51.0, abs=2.0)
    each = star_completeness(
        BINNED, flux, NOISE_E, fwhm_px=0.75, trail_px=length, trail_angle_rad=angle
    )
    exact = np.array(
        [
            float(
                SearchModel.build(
                    BINNED, fwhm_px=model_fwhm(0.75), trail_px=float(span), trail_angle_rad=float(a)
                ).completeness(flux, NOISE_E)
            )
            for span, a in zip(length[::5], angle[::5], strict=True)
        ]
    )
    assert float(np.mean(exact)) == pytest.approx(0.79, abs=0.02)
    assert float(np.mean(exact >= 0.9)) < 0.35
    assert np.all(each[::5] <= exact + 0.02)  # the grid errs low
    near = np.hypot(x - 1800.0, y - 900.0) < 120.0
    assert near.any()
    assert np.all(each[near] < 0.5)


def test_a_noise_map_gives_the_noise_at_any_position() -> None:
    """The map keeps one point per mesh cell, in counts of one pixel of the full frame."""
    # A binned search (2 x 2) of 64 by 32 sums, with a noise of 10 to 40 counts a sum across it.
    sums = np.repeat(np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32), 16, axis=1)
    noise = NoiseMap.sample(np.repeat(sums, 32, axis=0), scale=2, mesh_px=32)
    assert noise.step_px == 32.0
    assert noise.values.shape == (2, 4)
    assert noise.x0 == pytest.approx(16.5)  # the middle of the sum at column 8
    np.testing.assert_allclose(noise.at([0.0, 40.0, 1e4], [0.0, 0.0, 0.0]), [5.0, 10.0, 20.0])
    moved = noise.shifted(100.0, 50.0)
    np.testing.assert_allclose(moved.at([140.0], [50.0]), noise.at([40.0], [0.0]))
    frame, _ = injected_frame(100.0, 0, noisy_strip=True)
    found = detect_stars(frame, saturation_dn=1e9, options=DetectOptions(), e_per_adu=1.0)
    assert found.noise_map is not None
    quiet, loud = found.noise_map.at([100.0, 400.0], [100.0, 100.0])
    assert quiet == pytest.approx(NOISE_E, rel=0.1)
    assert loud == pytest.approx(2.0 * NOISE_E, rel=0.1)
    assert found.shifted(10.0, 20.0).noise_map is not None


def core_share(
    flux_sigma: float, spread: float, count: int | None = None
) -> tuple[float | None, float]:
    """`light_in_core` of a frame of stars drawn with the simulator's image, and their FWHM.

    The stars spread in flux as in a real field, so the brightest of them are the brightest by
    their light, not by the noise of their fits.
    """
    # The wide aperture of the growth curve and its ring reach 22 px, so the stars lie far apart.
    frame, _ = injected_frame(flux_sigma, 9, sim_psf=True, spacing=64, spread=spread)
    found = detect_stars(frame, saturation_dn=1e9, options=DetectOptions(), e_per_adu=1.0)
    fwhm = float(np.median(found.fwhm_px[found.reliable()]))
    if count is not None:
        found = found.select(np.arange(count))
    share = light_in_core(
        frame.astype(np.float32),
        found,
        origin=(0.0, 0.0),
        e_per_adu=1.0,
        saturation_dn=1e9,
        options=PhotometryOptions(),
    )
    return share, fwhm


def test_the_share_of_the_light_in_the_fit_keeps_the_model_honest_for_an_airy_image() -> None:
    """The simulator draws a star as an Airy pattern with a seeing halo: its fitted Gaussian holds
    0.91 of the light, and the rings hold the rest, below the threshold. A model that put all the
    light in the Gaussian overstates the share of these stars that the full search finds by 0.14
    at its 0.9 point (40 times the pixel noise), which would let a clear sky read clouds. With the
    share of the light that `light_in_core` measures, the bounds of the Gaussian cases hold.

    The share refers to the light within 12 px, as the zero point does, through the aperture
    correction of the growth curve, and it reads 0.91 on stars of 600 to 2,400 times the pixel
    noise. The correction takes stars with an SNR of 50 or more in the aperture of 5 px. Stars of
    200 to 400 times reach at most 38 there in a sky this bright, so the share keeps the light of
    that aperture and reads 0.925, high by the 1.5% that the aperture misses within 12 px.
    """
    core, fwhm = core_share(600.0, 4.0)
    assert core == pytest.approx(0.91, abs=0.01)
    faint, _ = core_share(200.0, 2.0)
    assert faint == pytest.approx(0.925, abs=0.01)
    assert core is not None
    for flux in (35.0, 40.0, 43.0):
        measured = found_share(flux, sim_psf=True)
        whole = model_share(FULL, flux, fwhm)
        honest = model_share(FULL, core * flux, fwhm)
        assert honest - 0.05 <= measured <= honest + 0.15, (flux, honest, measured)
        if flux == 40.0:
            assert whole - measured > 0.1  # 0.91 against 0.77 in the runs
    assert core_share(600.0, 4.0, count=2)[0] is None  # too few stars


def test_detections_keep_the_search_through_a_selection_and_a_shift() -> None:
    rng = np.random.default_rng(3)
    frame = 1000.0 + rng.normal(size=(128, 128)) * 10.0
    render_star(frame, 64.2, 63.7, 50_000.0, 1.2, 0.0)
    binned = detect_stars(
        frame, saturation_dn=1e9, options=DetectOptions(coarse_bin=2), trail=NO_ROTATION
    )
    full = detect_stars(frame, saturation_dn=1e9, options=DetectOptions(coarse_bin=2))
    assert binned.search == SearchSpec(5.0, 2, 2, relative=False)
    assert (binned.search.label, binned.search.binned) == ("binned2", True)
    assert full.search is not None
    assert (full.search.label, full.search.binned) == ("full", False)  # no trail model
    assert binned.select(np.arange(len(binned))).search == binned.search
    assert binned.shifted(10.0, 20.0).search == binned.search
