"""The seeing in a bright sky: the three methods against the simulator's truth.

The section "The seeing in a bright sky" of docs/research-notes.md holds the table. The quick tests
run the simulator's cheap optics (`gaussian`) for two windows of 10 s, and check what the table
shows: the old noise model reads `r0` far too small in daylight, the sky term brings the aperture's
centroid close and flags it `noisy`, and the Gaussian-weighted centroid stays on the truth. The
slow test reruns one case of the table with the wave optics and reproduces its numbers.
"""

from __future__ import annotations

import pytest

pytest.importorskip("seeingmon.drivers.sim", reason="the simulator needs SciPy (the fast extra)")

from seeingmon.drivers.sim.seeing_bias import BiasCase, CaseResult, exposure_for_sky, run_case
from seeingmon.profile import Profile, load_profile

DAYLIGHT_SKY = 4.2  # mag/arcsec^2, the simulator's sky near the pole above a Sun of +10 degrees
DARK_SKY = 20.5


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("asi294mm-gs250")


def quick(profile: Profile, sky: float) -> CaseResult:
    case = BiasCase(
        sky, exposure_for_sky(profile, sky), seed=3, windows=2, window_s=10.0, psf_mode="gaussian"
    )
    return run_case(profile, case)


@pytest.fixture(scope="module")
def daylight(profile: Profile) -> CaseResult:
    return quick(profile, DAYLIGHT_SKY)


@pytest.fixture(scope="module")
def dark(profile: Profile) -> CaseResult:
    return quick(profile, DARK_SKY)


def test_the_adaptive_exposure_puts_the_daylight_background_at_its_target(
    profile: Profile, daylight: CaseResult
) -> None:
    """1.23 ms, as in the detection estimate, and 0.31 of saturation with the camera's offset."""
    assert daylight.case.exposure_us == pytest.approx(1226, abs=2)
    for fraction in daylight.background_fraction:
        assert fraction == pytest.approx(0.307, abs=0.005)


def test_in_a_dark_sky_every_method_reads_the_truth(dark: CaseResult) -> None:
    """Two windows of 10 s scatter by about 4% around the truth, so the tolerance is 6%."""
    for method in ("a", "b", "c"):
        for ratio in dark.r0_ratio(method):
            assert ratio == pytest.approx(1.0, abs=0.06), method
    assert dark.noise_share("b") < 0.02
    assert all("noisy" not in flags for flags in dark.methods["b"].flags)
    assert all("noisy" not in flags for flags in dark.methods["c"].flags)


def test_in_daylight_the_old_model_reads_a_third_of_r0(daylight: CaseResult) -> None:
    """The aperture's noise is about 4 times the motion variance, and the old model left it in."""
    for ratio in daylight.r0_ratio("a"):
        assert ratio is not None
        assert 0.25 < ratio < 0.40
    assert daylight.noise_share("b") > 3.0


def test_in_daylight_the_sky_term_comes_close_and_the_window_is_noisy(
    daylight: CaseResult,
) -> None:
    """The model of the aperture reads about 5% low, so `r0` reads about 10% low."""
    for ratio in daylight.r0_ratio("b"):
        assert ratio is not None
        assert 0.75 < ratio < 1.05
    assert all("noisy" in flags for flags in daylight.methods["b"].flags)


def test_in_daylight_the_weighted_centroid_reads_what_it_reads_in_the_dark(
    daylight: CaseResult, dark: CaseResult
) -> None:
    """The same seed gives the same atmosphere, so the dark sky's reading of each window is the
    reference: the daylight noise moves it by under 1%."""
    for bright, reference in zip(daylight.r0_ratio("c"), dark.r0_ratio("c"), strict=True):
        assert bright is not None
        assert reference is not None
        assert bright == pytest.approx(reference, rel=0.01)
    assert daylight.noise_share("c") < 0.06
    assert all("noisy" not in flags for flags in daylight.methods["c"].flags)


@pytest.mark.slow
def test_the_daylight_row_of_the_table_for_its_first_seed(profile: Profile) -> None:
    """The daylight row of the table at an `r0` of 10 cm, seed 1: three windows of 60 s with the
    wave optics. The run is the same on every machine, so the numbers come back to rounding; the
    tolerances leave room for the last bits of the FFTs of another platform, 1% for the methods
    whose noise is small against the motion and 2% for the aperture's, which the noise rules."""
    case = BiasCase(DAYLIGHT_SKY, exposure_for_sky(profile, DAYLIGHT_SKY), seed=1, windows=3)
    result = run_case(profile, case)
    assert result.truth_r0_cm == pytest.approx(10.0)
    assert result.star_snr == pytest.approx((8.73, 8.69, 8.68), abs=0.02)
    assert result.methods["a"].r0_cm == pytest.approx((3.345, 3.317, 3.271), rel=0.01)
    assert result.methods["b"].r0_cm == pytest.approx((8.818, 8.938, 9.047), rel=0.02)
    assert result.methods["c"].r0_cm == pytest.approx((10.095, 10.104, 10.076), rel=0.01)
    assert result.noise_share("b") == pytest.approx(4.2, abs=0.2)
    assert all("noisy" in flags for flags in result.methods["b"].flags)
    assert all("noisy" not in flags for flags in result.methods["c"].flags)
