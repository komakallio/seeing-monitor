"""The detection estimate: the SNR of Polaris in one fast frame against the Sun's elevation.

The section "Polaris in a bright sky" of docs/research-notes.md holds the tables, the crossing (or
its absence), and the search limit. These tests rebuild them from the reference profile and the
default settings, so a change to either shows up here.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from seeingmon.config import load_config
from seeingmon.drivers.sim.detection import DETECTION_SNR, DetectionModel
from seeingmon.drivers.sim.params import ARCSEC_PER_RAD, SimParams
from seeingmon.drivers.sim.sky import airmass, flux_factor
from seeingmon.fastpath.config import FastPathConfig
from seeingmon.fastpath.kernel import KernelParams
from seeingmon.fastpath.matched import matched_filter
from seeingmon.profile import load_profile
from seeingmon.scheduler.config import NO_SUN_LIMIT_DEG, SchedulerConfig


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> DetectionModel:
    """The model of the research notes: the reference profile and the default settings."""
    profile = load_profile("asi294mm-gs250")
    absent = tmp_path_factory.mktemp("config") / "absent.toml"
    config = load_config(local_file=absent, env={})
    fast = config.section("scheduler", SchedulerConfig).fast
    fastpath = config.section("fastpath", FastPathConfig)
    mode = profile.fast_mode.mode
    airy = profile.airy_fwhm_px(mode)
    return DetectionModel.for_simulator(
        SimParams.from_profile(profile, mode),
        gain=fast.gain,
        max_exposure_us=float(fast.exposure_us),
        min_exposure_us=float(profile.limits.exposure_us_range[0]),
        aperture_diameter_px=max(fastpath.aperture_min_px, fastpath.aperture_airy_widths * airy),
        filter_fwhms_px=tuple(widths * airy for widths in fastpath.matched_fwhm_airy_widths),
    )


def notes_section(repo_root: Path) -> str:
    text = (repo_root / "docs" / "research-notes.md").read_text(encoding="utf-8")
    start = text.index("## Polaris in a bright sky")
    return text[start : text.index("\n## ", start + 1)]


MINUS = chr(0x2212)  # the tables of the documentation write negative numbers with it


def elevation(label: str) -> list[float]:
    """The Sun elevations of a row label such as `+3`, `0`, `-18`, or `+10 to +60`."""
    numbers = [float(part.replace(MINUS, "-")) for part in label.split(" to ")]
    return numbers if len(numbers) == 1 else [numbers[0], sum(numbers) / 2, numbers[1]]


def bisect(function: object, low: float, high: float) -> float:
    """Where a falling function crosses `DETECTION_SNR` between `low` and `high`."""
    assert callable(function)
    assert function(low) > DETECTION_SNR > function(high)
    while high - low > 1e-5:
        middle = 0.5 * (low + high)
        if function(middle) >= DETECTION_SNR:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def test_the_apertures_are_the_fast_paths(model: DetectionModel) -> None:
    assert model.aperture_diameter_px == pytest.approx(16.0, abs=0.01)  # 12 Airy FWHM of 1.333 px
    # The kernel's own area of a centered aperture. The model averages over the star's position
    # within a pixel, which moves the area of a soft edge by a few tenths of a percent.
    kernel = KernelParams(aperture_diameter_px=model.aperture_diameter_px)
    assert model.centroid.area_px == pytest.approx(kernel.area_px2, rel=0.005)
    assert model.centroid.flux_fraction == pytest.approx(0.970, abs=0.0005)
    # The matched filters of the kernel: Gaussians of 1, 2, and 4 Airy FWHM, the default of
    # [fastpath].
    assert model.airy_fwhm_px == pytest.approx(1.333, abs=0.001)
    assert model.filter_widths_px == pytest.approx((1.333, 2.667, 5.334), abs=0.001)
    assert DetectionModel(model.params).filter_widths_px == pytest.approx(model.filter_widths_px)


def test_the_image_of_polaris_spreads_over_6_pixels(model: DetectionModel) -> None:
    """`1 / sum(P^2)` of the simulator's image is 5.9 px^2, and the first filter's response 6.0."""
    stamps, _ = model._stamps()
    area = float(np.mean([1.0 / np.sum(stamp**2) for stamp in stamps]))
    assert area == pytest.approx(5.9, abs=0.05)
    assert model.image_area_px2 == pytest.approx(area)
    assert model.matched[0].effective_area_px2 == pytest.approx(6.0, abs=0.05)
    first = matched_filter(model.filter_widths_px[0])
    assert first.effective_area_px2 == pytest.approx(5.1, abs=0.05)


def test_the_exposure_keeps_the_background_at_the_target(model: DetectionModel) -> None:
    dark = model.row(-18.0)
    assert dark.exposure_us == pytest.approx(2000.0)  # the cap
    day = model.row(30.0)
    assert day.exposure_us < 2000.0
    assert day.background_fraction == pytest.approx(0.3, abs=1e-9)
    glare = model.at_sky(-3.0)  # far brighter than any daylight sky near the pole
    assert glare.exposure_us == pytest.approx(32.0)  # the profile's shortest exposure
    assert glare.background_fraction > 0.3


def test_the_table_of_the_research_notes(model: DetectionModel, repo_root: Path) -> None:
    section = notes_section(repo_root)
    start = section.index("| Sun (°) |")
    table = section[start : section.index("\n\n", start)]
    rows = [
        line
        for line in table.splitlines()
        if re.match(r"^\| (\+\d+( to \+\d+)?|0|" + MINUS + r"\d+) \|", line)
    ]
    assert len(rows) == 17
    for line in rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        sky, exposure_ms, fraction, star, best, matched, centroid = (
            float(cell.replace(",", "")) for cell in cells[1:]
        )
        for sun in elevation(cells[0]):
            row = model.row(sun)
            # Each number matches to half of its last printed digit. The Polaris column is
            # rounded to tens of electrons, so its tolerance is 5 e-.
            assert row.sky_mag_arcsec2 == pytest.approx(sky, abs=0.0051), line
            assert row.exposure_us / 1000 == pytest.approx(exposure_ms, abs=0.0051), line
            assert row.background_fraction == pytest.approx(fraction, abs=0.0051), line
            assert row.star_e == pytest.approx(star, abs=5.1), line
            assert row.snr_best == pytest.approx(best, abs=0.051), line
            assert row.snr_matched == pytest.approx(matched, abs=0.051), line
            assert row.snr_centroid == pytest.approx(centroid, abs=0.051), line


def test_the_physical_limit_of_the_daylight_frame(model: DetectionModel) -> None:
    """The best weighting of the pixels, `P / (v + F P)`, gives 41.5, and the filters 41.3.

    Without the star's own photons, the sky alone allows `F / sqrt(v / sum(P^2))`, which is 50.
    """
    row = model.row(30.0)
    sensor = model.params.sensor_at(model.gain)
    pixel_var = row.background_e + sensor.read_noise_e**2 + sensor.e_per_adu**2 / 12
    assert pixel_var == pytest.approx(4_310, abs=5)
    exposure_s = row.exposure_us * 1e-6
    rms = model.scintillation.index(exposure_s, airmass(model.zenith_angle_deg))
    flux = float(flux_factor(rms, 0.0)) * row.star_e
    assert flux == pytest.approx(7_980, abs=5)
    stamps, _ = model._stamps()
    best = np.mean([math.sqrt(np.sum((flux * p) ** 2 / (pixel_var + flux * p))) for p in stamps])
    sky_only = np.mean([flux * math.sqrt(np.sum(p**2) / pixel_var) for p in stamps])
    assert best == pytest.approx(41.5, abs=0.05)
    assert row.snr_best == pytest.approx(best)
    assert sky_only == pytest.approx(50.2, abs=0.05)
    assert row.snr_matched == pytest.approx(41.3, abs=0.05)
    assert row.snr_filters[0] == row.snr_matched  # the first filter fits the simulator's image
    assert row.snr_matched / best > 0.99  # the matched filter is nearly the best weighting
    # In a dark sky the star's photons dominate. The best weighting gives 0.96 of `sqrt(F)`, the
    # widest filter, which takes in the halo, 0.95, and the first filter 0.83.
    dark = model.row(-18.0)
    rms = model.scintillation.index(dark.exposure_us * 1e-6, airmass(model.zenith_angle_deg))
    root = math.sqrt(float(flux_factor(rms, 0.0)) * dark.star_e)
    assert dark.snr_best / root == pytest.approx(0.96, abs=0.005)
    assert dark.snr_matched / root == pytest.approx(0.95, abs=0.005)
    assert dark.snr_filters[0] / root == pytest.approx(0.83, abs=0.005)
    # No filter passes the best weighting, in any sky.
    for sun in (-18.0, -4.0, 0.0, 5.0, 30.0):
        check = model.row(sun)
        assert max(check.snr_filters) == check.snr_matched <= check.snr_best, sun


def test_no_crossing_and_no_search_limit(model: DetectionModel, repo_root: Path) -> None:
    """The matched SNR stays above 10 in full daylight, so the search has no Sun limit."""
    assert model.crossing_deg() is None
    assert min(model.row(float(sun)).snr_matched for sun in range(-18, 91)) == pytest.approx(
        41.3, abs=0.05
    )
    # Without a crossing, no limit. A crossing would give the crossing plus a few degrees.
    limit = NO_SUN_LIMIT_DEG
    section = notes_section(repo_root)
    assert f"`scheduler.search.max_sun_elevation_deg` is **{limit:.0f}°**" in section
    assert SchedulerConfig().search.max_sun_elevation_deg == limit
    architecture = (repo_root / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert f"The default `max_sun_elevation_deg` of {limit:.0f} degrees means no limit" in (
        architecture
    )


def test_the_matched_snr_falls_to_10_only_in_a_far_brighter_sky(model: DetectionModel) -> None:
    """At 2.53 mag/arcsec^2: 1.67 mag brighter than the model's daylight, and 0.67 mag brighter
    than the brightest daylight sky measured near the pole."""
    # The SNR falls as the sky brightens, which is as its magnitude falls.
    sky = -bisect(lambda brightness: model.at_sky(-brightness).snr_matched, -4.2, -1.0)
    assert sky == pytest.approx(2.53, abs=0.005)
    found = model.crossing_sky_mag_arcsec2()
    assert found is not None
    assert found == pytest.approx(sky, abs=1e-3)
    assert model.at_sky(sky).exposure_us / 1000 == pytest.approx(0.26, abs=0.005)


def test_the_daylight_skies_that_were_measured(model: DetectionModel) -> None:
    # The brightest and the darkest sky that Nickel and Calderwood measured near the pole's angle
    # from the Sun, and the median that the simulator uses.
    assert model.at_sky(3.2).snr_matched == pytest.approx(17.9, abs=0.05)
    assert model.at_sky(4.2).snr_matched == pytest.approx(41.3, abs=0.05)
    assert model.at_sky(4.7).snr_matched == pytest.approx(60.8, abs=0.05)
    assert model.at_sky(3.2).exposure_us / 1000 == pytest.approx(0.49, abs=0.005)
    assert model.at_sky(4.7).exposure_us / 1000 == pytest.approx(1.94, abs=0.005)


def test_what_the_centroid_aperture_loses(model: DetectionModel) -> None:
    """In daylight the aperture gives a fifth of the matched SNR, and it crosses 10 at +8.9°,
    where the first version of the estimate put the search limit. In a dark sky, where the
    star's photons dominate, it gives more than the first filter, and less than the widest."""
    day, dark = model.row(30.0), model.row(-18.0)
    assert day.snr_centroid == pytest.approx(8.3, abs=0.05)
    assert day.snr_centroid / day.snr_matched == pytest.approx(0.20, abs=0.005)
    assert dark.snr_centroid == pytest.approx(106.6, abs=0.05)
    assert dark.snr_filters[0] == pytest.approx(95.2, abs=0.05)
    assert dark.snr_matched == pytest.approx(109.5, abs=0.05)
    crossing = bisect(lambda sun: model.row(sun).snr_centroid, 0.0, 30.0)
    assert crossing == pytest.approx(8.9, abs=0.05)
    assert model.row(crossing).sky_mag_arcsec2 == pytest.approx(4.40, abs=0.005)


def test_the_crossing_reports_a_star_that_fades_or_never_shows(
    model: DetectionModel,
) -> None:
    params = model.params
    assert DetectionModel.for_simulator(params, polaris_mag=-1.0).crossing_deg() is None
    faint = DetectionModel.for_simulator(params, polaris_mag=6.0)
    crossing = faint.crossing_deg()
    assert crossing is not None
    assert -18.0 < crossing < 10.0  # a star 4 mag fainter fades in the twilight
    assert faint.row(crossing).snr_matched == pytest.approx(DETECTION_SNR, abs=1e-3)
    never = DetectionModel.for_simulator(params, polaris_mag=14.0)
    assert never.crossing_deg() == -18.0  # never detectable in this range
    assert math.isnan(model.at_sky(4.2).sun_elevation_deg)
    with pytest.raises(ValueError, match="exposures"):
        DetectionModel(params, min_exposure_us=3000.0)
    with pytest.raises(ValueError, match="target_background_fraction"):
        DetectionModel(params, target_background_fraction=0.0)
    with pytest.raises(ValueError, match="blur_fwhm_arcsec"):
        DetectionModel(params, blur_fwhm_arcsec=-1.0)
    with pytest.raises(ValueError, match="filter_fwhms_px"):
        DetectionModel(params, filter_fwhms_px=())


def image_rows(repo_root: Path) -> list[list[float]]:
    """The rows of the table of image sizes in the research notes, as numbers."""
    section = notes_section(repo_root)
    start = section.index("| Blur (FWHM, arcsec) |")
    table = section[start : section.index("\n\n", start)].splitlines()[2:]
    return [
        [float(cell.strip().replace(",", "")) for cell in line.strip("|").split("|")]
        for line in table
    ]


def test_the_table_of_image_sizes(model: DetectionModel, repo_root: Path) -> None:
    """The SNR of the median daylight frame against a Gaussian blur of the simulator's image."""
    rows = image_rows(repo_root)
    assert [row[0] for row in rows] == [0.0, 3.0, 5.0, 7.0, 10.0, 13.0]
    arcsec2_per_px2 = (model.params.pixel_rad * ARCSEC_PER_RAD) ** 2
    for blur, area, best, matched, first, centroid, sky in rows:
        blurred = DetectionModel.for_simulator(
            model.params,
            gain=model.gain,
            max_exposure_us=model.max_exposure_us,
            min_exposure_us=model.min_exposure_us,
            aperture_diameter_px=model.aperture_diameter_px,
            filter_fwhms_px=model.filter_fwhms_px,
            blur_fwhm_arcsec=blur,
        )
        row = blurred.row(30.0)
        # Each number matches to half of its last printed digit.
        assert blurred.image_area_px2 * arcsec2_per_px2 == pytest.approx(area, abs=0.51), blur
        assert row.snr_best == pytest.approx(best, abs=0.051), blur
        assert row.snr_matched == pytest.approx(matched, abs=0.051), blur
        assert row.snr_filters[0] == pytest.approx(first, abs=0.051), blur
        assert row.snr_centroid == pytest.approx(centroid, abs=0.051), blur
        crossing = blurred.crossing_sky_mag_arcsec2()
        assert crossing is not None
        assert crossing == pytest.approx(sky, abs=0.0051), blur
        # The bank follows the best weighting within 5%, where the first filter alone falls off.
        assert row.snr_matched / row.snr_best > 0.95, blur
        # No crossing of 10 between -18 and +90 degrees in the model's sky, even at 13 arcsec.
        assert blurred.crossing_deg() is None, blur


def test_the_image_of_the_recordings_matches_a_blur_of_10_arcsec(model: DetectionModel) -> None:
    """The owner's recordings give `1 / sum(P^2)` of 276 to 286 arcsec^2 (research notes).

    The simulator's image blurred by 10 arcsec covers 284 arcsec^2, so that row of the table
    stands for the real image.
    """
    blurred = DetectionModel.for_simulator(
        model.params,
        gain=model.gain,
        max_exposure_us=model.max_exposure_us,
        min_exposure_us=model.min_exposure_us,
        aperture_diameter_px=model.aperture_diameter_px,
        blur_fwhm_arcsec=10.0,
    )
    arcsec2_per_px2 = (model.params.pixel_rad * ARCSEC_PER_RAD) ** 2
    assert 276.0 <= blurred.image_area_px2 * arcsec2_per_px2 <= 286.0
    assert model.image_area_px2 * arcsec2_per_px2 == pytest.approx(21.5, abs=0.05)
