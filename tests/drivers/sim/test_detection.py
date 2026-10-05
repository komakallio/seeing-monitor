"""The detection estimate: the SNR of Polaris in one fast frame against the sun's elevation.

The section "Polaris in a bright sky" of docs/research-notes.md holds the table, the crossing, and
the search limit. These tests rebuild them from the reference profile and the default settings, so
a change to either shows up here.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from seeingmon.config import load_config
from seeingmon.drivers.sim.detection import DETECTION_SNR, DetectionModel
from seeingmon.drivers.sim.params import SimParams
from seeingmon.fastpath.config import FastPathConfig
from seeingmon.fastpath.kernel import KernelParams
from seeingmon.profile import load_profile
from seeingmon.scheduler.config import SchedulerConfig

SEARCH_MARGIN_DEG = 3.0  # the margin above the crossing that gives the search limit


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> DetectionModel:
    """The model of the research notes: the reference profile and the default settings."""
    profile = load_profile("asi294mm-gs250")
    absent = tmp_path_factory.mktemp("config") / "absent.toml"
    config = load_config(local_file=absent, env={})
    fast = config.section("scheduler", SchedulerConfig).fast
    fastpath = config.section("fastpath", FastPathConfig)
    mode = profile.fast_mode.mode
    diameter = max(
        fastpath.aperture_min_px, fastpath.aperture_airy_widths * profile.airy_fwhm_px(mode)
    )
    return DetectionModel.for_simulator(
        SimParams.from_profile(profile, mode),
        gain=fast.gain,
        max_exposure_us=float(fast.exposure_us),
        min_exposure_us=float(profile.limits.exposure_us_range[0]),
        aperture_diameter_px=diameter,
    )


def notes_section(repo_root: Path) -> str:
    text = (repo_root / "docs" / "research-notes.md").read_text(encoding="utf-8")
    start = text.index("## Polaris in a bright sky")
    return text[start : text.index("\n## ", start + 1)]


MINUS = chr(0x2212)  # the tables of the documentation write negative numbers with it


def elevation(label: str) -> list[float]:
    """The sun elevations of a row label such as `+3`, `0`, `-18`, or `+10 to +60`."""
    numbers = [float(part.replace(MINUS, "-")) for part in label.split(" to ")]
    return numbers if len(numbers) == 1 else [numbers[0], sum(numbers) / 2, numbers[1]]


def mean_frame_crossing_deg(model: DetectionModel) -> float:
    """The sun elevation at which the SNR of the mean frame falls to `DETECTION_SNR`."""
    low, high = 0.0, 30.0  # the mean-frame SNR is above the threshold at 0 and below it at 30
    assert model.row(low).snr_mean > DETECTION_SNR > model.row(high).snr_mean
    while high - low > 1e-4:
        middle = 0.5 * (low + high)
        if model.row(middle).snr_mean >= DETECTION_SNR:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def test_the_aperture_is_the_fast_paths(model: DetectionModel) -> None:
    assert model.aperture_diameter_px == pytest.approx(16.0, abs=0.01)  # 12 Airy FWHM of 1.333 px
    # The kernel's own area of a centered aperture. The model averages over the star's position
    # within a pixel, which moves the area of a soft edge by a few tenths of a percent.
    kernel = KernelParams(aperture_diameter_px=model.aperture_diameter_px)
    assert model.fast.area_px == pytest.approx(kernel.area_px2, rel=0.005)
    assert model.fast.flux_fraction == pytest.approx(0.970, abs=0.0005)


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
    rows = [
        line
        for line in notes_section(repo_root).splitlines()
        if re.match(r"^\| (\+\d+( to \+\d+)?|0|" + MINUS + r"\d+) \|", line)
    ]
    assert len(rows) >= 15
    for line in rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        sky, exposure_ms, fraction, star, mean, median, matched = (
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
            assert row.snr_mean == pytest.approx(mean, abs=0.051), line
            assert row.snr_median == pytest.approx(median, abs=0.051), line
            assert row.snr_matched == pytest.approx(matched, abs=0.051), line


def test_the_crossing_and_the_search_limit(model: DetectionModel, repo_root: Path) -> None:
    crossing = model.crossing_deg()
    assert crossing is not None  # Polaris fades below the threshold in the model's daylight
    assert crossing == pytest.approx(8.9, abs=0.05)  # degrees
    at = model.row(crossing)
    assert at.snr_median == pytest.approx(DETECTION_SNR, abs=1e-3)
    assert at.sky_mag_arcsec2 == pytest.approx(4.40, abs=0.005)
    assert at.exposure_us / 1000 == pytest.approx(1.48, abs=0.005)
    section = notes_section(repo_root)
    # The notes print the crossings to 0.1 degree, so each matches to 0.05 degree.
    match = re.search(r"falls to 10 at a Sun elevation of \*\*\+(\d+\.\d)°\*\*", section)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(crossing, abs=0.05)
    mean_crossing = mean_frame_crossing_deg(model)
    assert model.row(mean_crossing).snr_mean == pytest.approx(DETECTION_SNR, abs=1e-3)
    match = re.search(r"The mean frame crosses at \+(\d+\.\d)°", section)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(mean_crossing, abs=0.05)
    limit = round(crossing + SEARCH_MARGIN_DEG)
    assert f"`scheduler.search.max_sun_elevation_deg` of **{limit}°**" in section
    design = (repo_root / "docs" / "visibility.md").read_text(encoding="utf-8")
    assert f"| `scheduler.search.max_sun_elevation_deg` | {limit:.1f} |" in design


def test_the_daylight_sky_decides_the_detection(model: DetectionModel) -> None:
    # The brightest and the darkest sky that Nickel and Calderwood measured near the pole's angle
    # from the sun, and the median that the simulator uses.
    assert model.at_sky(3.2).snr_median == pytest.approx(3.2, abs=0.05)
    assert model.at_sky(4.2).snr_median == pytest.approx(8.3, abs=0.05)
    assert model.at_sky(4.7).snr_median == pytest.approx(13.2, abs=0.05)
    assert model.at_sky(3.2).exposure_us / 1000 == pytest.approx(0.49, abs=0.005)
    assert model.at_sky(4.7).exposure_us / 1000 == pytest.approx(1.94, abs=0.005)


def test_a_matched_aperture_gains_in_a_bright_sky_only(model: DetectionModel) -> None:
    matched = model.matched
    assert matched.radius_px == pytest.approx(1.0, abs=0.01)
    assert matched.area_px == pytest.approx(3.4, abs=0.05)
    assert matched.flux_fraction == pytest.approx(0.62, abs=0.005)
    day, dark = model.row(30.0), model.row(-18.0)
    assert day.snr_matched / day.snr_median == pytest.approx(4.2, abs=0.05)
    assert dark.snr_matched / dark.snr_median == pytest.approx(0.84, abs=0.005)
    assert model.at_sky(3.2).snr_matched == pytest.approx(14.9, abs=0.05)


def test_the_crossing_reports_a_star_that_never_fades_or_never_shows(
    model: DetectionModel,
) -> None:
    params = model.params
    bright = DetectionModel.for_simulator(params, polaris_mag=-1.0)
    assert bright.crossing_deg() is None  # detectable in full daylight: no search limit
    faint = DetectionModel.for_simulator(params, polaris_mag=14.0)
    assert faint.crossing_deg() == -18.0  # never detectable in this range
    assert math.isnan(model.at_sky(4.2).sun_elevation_deg)
    with pytest.raises(ValueError, match="exposures"):
        DetectionModel(params, min_exposure_us=3000.0)
    with pytest.raises(ValueError, match="target_background_fraction"):
        DetectionModel(params, target_background_fraction=0.0)
