"""The readout-mode parameters of the simulator, and their agreement with the profile."""

from __future__ import annotations

import dataclasses
import math

import pytest

from seeingmon.drivers.sim.params import GainPoint, SimParams, reference_modes
from seeingmon.frames import Roi
from seeingmon.profile import Profile, load_profile


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("asi294mm-gs250")


def test_plate_scales_match_the_research_notes() -> None:
    assert SimParams.reference("bin1").plate_scale_arcsec_per_px == pytest.approx(1.910, abs=0.001)
    assert SimParams.reference("bin2").plate_scale_arcsec_per_px == pytest.approx(3.820, abs=0.001)


def test_frame_periods_match_the_research_notes() -> None:
    # "Bin1, 128 rows: 6.5 ms + 128 x 37.6 us = 11.3 ms, or 88 fps."
    bin1 = SimParams.reference("bin1")
    assert bin1.readout_time_s(128) == pytest.approx(11.3e-3, abs=0.02e-3)
    # "Bin2, 64 rows: 1.4 ms + 64 x 21.3 us = 2.8 ms, or 360 fps."
    bin2 = SimParams.reference("bin2")
    assert 1 / bin2.readout_time_s(64) == pytest.approx(360, rel=0.02)


def test_the_reference_equals_the_reference_profile(profile: Profile) -> None:
    for name in ("bin1", "bin2"):
        explicit = SimParams.reference(name)
        loaded = SimParams.from_profile(profile, name)
        assert explicit.high_speed is not None
        assert loaded.high_speed is not None
        for left, right in ((explicit, loaded), (explicit.high_speed, loaded.high_speed)):
            for field in dataclasses.fields(SimParams):
                if field.name == "high_speed":
                    continue
                a = getattr(left, field.name)
                b = getattr(right, field.name)
                if isinstance(a, float):
                    assert a == pytest.approx(b, rel=1e-12), field.name
                elif field.name == "dark_table":
                    assert len(a) == len(b)
                    for (ta, ra), (tb, rb) in zip(a, b, strict=True):
                        assert (ta, ra) == pytest.approx((tb, rb))
                else:
                    assert a == b, field.name


@pytest.mark.parametrize("mode", ["bin1", "bin2"])
@pytest.mark.parametrize("high_speed", [False, True])
def test_sensor_numbers_match_the_profile_at_every_gain(
    profile: Profile, mode: str, high_speed: bool
) -> None:
    params = SimParams.from_profile(profile, mode).effective(high_speed=high_speed)
    readout = profile.mode(mode, high_speed=high_speed)
    for gain in (0, 1, 50, 107, 108, 109, 118, 119, 120, 121, 200, 269, 270, 300, 450, 570):
        sensor = params.sensor_at(gain)
        assert sensor.e_per_adu == pytest.approx(profile.e_per_adu(readout, gain), rel=1e-12)
        assert sensor.read_noise_e == pytest.approx(profile.read_noise_e(readout, gain), rel=1e-12)
        assert sensor.full_well_e == pytest.approx(profile.full_well_e(readout, gain), rel=1e-12)


def test_dark_current_follows_the_profile_prior(profile: Profile) -> None:
    params = SimParams.from_profile(profile, "bin1")
    for temperature in (-25.0, -10.0, 5.0, 20.0, 27.0, 35.0):
        assert params.dark_rate_e_per_s(temperature) == pytest.approx(
            profile.dark_current_e_per_s_per_px(temperature), rel=1e-12
        )
    # Binned pixels sum four native pixels.
    bin2 = SimParams.from_profile(profile, "bin2")
    assert bin2.dark_rate_e_per_s(20.0) == pytest.approx(4 * 0.2)


def test_dark_current_doubles_every_six_degrees_without_a_table() -> None:
    params = SimParams()
    assert params.dark_rate_e_per_s(20.0) == pytest.approx(0.2)
    assert params.dark_rate_e_per_s(26.0) == pytest.approx(0.4)
    assert params.dark_rate_e_per_s(8.0) == pytest.approx(0.05)


def test_photon_rates() -> None:
    params = SimParams.reference("bin1")
    assert params.mag0_rate_e_per_s() == pytest.approx(4.6e7)
    # A magnitude-8 star gives 2.9e5 electrons in 10 s ("Photometric precision").
    assert params.mag0_rate_e_per_s() * 10 ** (-0.4 * 8) * 10 == pytest.approx(2.9e5, rel=0.02)
    wide = dataclasses.replace(params, aperture_m=0.100)
    assert wide.mag0_rate_e_per_s() == pytest.approx(4 * 4.6e7)
    # The sky at 21 mag/arcsec2: 4.6e7 * 10^(-8.4) electrons per second per square arcsecond.
    sky = params.sky_rate_e_per_s_px(21.0)
    assert sky == pytest.approx(4.6e7 * 10 ** (-8.4) * params.plate_scale_arcsec_per_px**2)


def test_roi_rounding_follows_the_vendor_rules() -> None:
    params = SimParams.reference("bin1")
    assert params.normalize_roi(Roi(8200, 5600, 131, 125)) == Roi(8160, 5520, 128, 124)
    assert params.normalize_roi(None) == Roi(0, 0, 8288, 5644)
    assert params.normalize_roi(Roi(5, 5, 3, 1)) == Roi(5, 5, 8, 2)
    with pytest.raises(ValueError, match="larger than the frame"):
        params.normalize_roi(Roi(0, 0, 9000, 8))


def test_the_high_speed_variant_is_separate() -> None:
    params = SimParams.reference("bin1")
    fast = params.effective(high_speed=True)
    assert (fast.adc_bits, fast.row_time_s, fast.frame_overhead_s) == (10, 30.1e-6, 5.0e-3)
    assert params.effective(high_speed=False) is params
    with pytest.raises(ValueError, match="no high-speed"):
        SimParams().effective(high_speed=True)


def test_validation() -> None:
    with pytest.raises(ValueError, match="gain_points"):
        SimParams(gain_points=(GainPoint(10, 1.0, 1.0), GainPoint(5, 1.0, 1.0)))
    with pytest.raises(ValueError, match="adc_bits"):
        SimParams(adc_bits=4)
    with pytest.raises(ValueError, match="dark_table"):
        SimParams(dark_table=((20.0, 0.2),))
    with pytest.raises(ValueError, match="unknown reference mode"):
        SimParams.reference("bin3")
    assert set(reference_modes()) == {"bin1", "bin2"}
    assert math.isfinite(SimParams().pixel_rad)
