"""The fit of the SQM-LE offset, the dome loss, and the altitude term on synthetic readings."""

from __future__ import annotations

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.records.reference import ReferenceRecord
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.survey import sqm_fit

T0 = iso_to_utc_ns("2026-10-01T20:00:00Z")
CAMERA_ALTITUDE = 60.0  # a synthetic site at latitude 60
OFFSET, DOME, K = 0.15, 0.30, -0.25  # the truth that the fit must find
SQM_ALTITUDE = 45.0


def sky_record(t_utc_ns: int, v_mag: float | None) -> SkyQualityRecord:
    return SkyQualityRecord(
        station_id="test-station",
        t_utc_ns=t_utc_ns,
        profile_id="profile-1",
        provenance={"algo": "sky-1"},
        sky_mag_arcsec2_v=v_mag,
        n_stars_used=60,
    )


def reading(t_utc_ns: int, value: float, *, source: str, altitude: float | None) -> ReferenceRecord:
    return ReferenceRecord(
        station_id="test-station",
        t_utc_ns=t_utc_ns,
        profile_id="profile-1",
        provenance={"algo": "sqm-1"},
        instrument="sqm-le" if source == "fixed" else "sqm-l",
        source=source,
        value_mag_arcsec2=value,
        altitude_deg=altitude,
    )


def synthetic_night(
    *,
    seed: int = 1,
    n_fixed: int = 120,
    manual_altitudes: tuple[float, ...] = (30.0, 45.0, 60.0, 75.0),
    n_manual_each: int = 10,
    noise_mag: float = 0.03,
    outlier_fraction: float = 0.0,
    camera_altitude: float = CAMERA_ALTITUDE,
) -> tuple[list[ReferenceRecord], list[SkyQualityRecord]]:
    """Readings of an SQM-LE behind a dome and handheld readings, with the survey's own values."""
    rng = np.random.default_rng(seed)
    references: list[ReferenceRecord] = []
    sky: list[SkyQualityRecord] = []
    x_camera = sqm_fit.airmass(camera_altitude)
    total = n_fixed + n_manual_each * len(manual_altitudes)
    times = T0 + np.arange(total) * 300 * NS_PER_S  # every 5 minutes
    manual_plan = [alt for alt in manual_altitudes for _ in range(n_manual_each)]
    order = rng.permutation(total)
    for slot, t in zip(order, times, strict=True):
        ours = 20.8 + 0.4 * np.sin(2.0 * np.pi * (int(t) - T0) / (4 * 3600 * NS_PER_S))
        sky.append(sky_record(int(t), float(ours)))
        if slot < n_fixed:
            source, altitude, dome = "fixed", SQM_ALTITUDE, DOME
        else:
            source, altitude, dome = "manual", manual_plan[slot - n_fixed], 0.0
        truth = ours + OFFSET + dome + K * (sqm_fit.airmass(altitude) - x_camera)
        value = float(truth + rng.normal(0.0, noise_mag))
        if rng.random() < outlier_fraction:
            value -= 1.2  # the moon came up, or a cloud drifted over the dome
        references.append(reading(int(t) + 20 * NS_PER_S, value, source=source, altitude=altitude))
    return references, sky


def test_the_airmass_follows_kasten_and_young() -> None:
    assert sqm_fit.airmass(90.0) == pytest.approx(1.0, abs=0.001)
    assert sqm_fit.airmass(60.0) == pytest.approx(1.154, abs=0.002)
    assert sqm_fit.airmass(45.0) == pytest.approx(1.413, abs=0.002)
    assert sqm_fit.airmass(30.0) == pytest.approx(1.995, abs=0.003)
    assert sqm_fit.airmass(10.0) == pytest.approx(5.6, abs=0.1)
    assert sqm_fit.airmass(-5.0) == sqm_fit.airmass(0.0)  # the horizon, not a negative altitude
    altitudes = np.linspace(5.0, 90.0, 40)
    assert np.all(np.diff([sqm_fit.airmass(float(h)) for h in altitudes]) < 0.0)


def test_the_fit_finds_the_offset_the_dome_loss_and_the_altitude_term() -> None:
    references, sky = synthetic_night()
    fit = sqm_fit.fit_sqm_offset(references, sky, camera_altitude_deg=CAMERA_ALTITUDE)
    assert fit is not None
    assert (fit.n_fixed, fit.n_manual) == (120, 40)
    assert 155 <= fit.n_used <= 160  # the 3-sigma clip may take a pair or two of 160
    assert fit.offset_mag == pytest.approx(OFFSET, abs=3.0 * fit.offset_error_mag + 0.01)
    assert fit.dome_loss_mag == pytest.approx(DOME, abs=0.04)
    assert fit.altitude_mag_per_airmass == pytest.approx(K, abs=0.05)
    assert fit.rms_mag == pytest.approx(0.03, rel=0.2)
    # The sampling errors are a few hundredths of a magnitude at most.
    assert fit.offset_error_mag < 0.02
    assert fit.dome_loss_error_mag is not None
    assert fit.dome_loss_error_mag < 0.02


def test_fixed_readings_alone_give_the_offset_and_the_dome_together() -> None:
    references, sky = synthetic_night(manual_altitudes=(), n_fixed=150)
    fit = sqm_fit.fit_sqm_offset(references, sky, camera_altitude_deg=CAMERA_ALTITUDE)
    assert fit is not None
    assert fit.n_manual == 0
    assert fit.dome_loss_mag is None  # a constant dome loss cannot be told from the band offset
    assert fit.altitude_mag_per_airmass is None  # one altitude fixes no slope
    expected = (
        OFFSET + DOME + K * (sqm_fit.airmass(SQM_ALTITUDE) - sqm_fit.airmass(CAMERA_ALTITUDE))
    )
    assert fit.offset_mag == pytest.approx(expected, abs=0.02)


def test_handheld_readings_alone_fix_the_altitude_term_but_no_dome() -> None:
    references, sky = synthetic_night(
        n_fixed=0, manual_altitudes=(20.0, 40.0, 60.0, 80.0), n_manual_each=20
    )
    fit = sqm_fit.fit_sqm_offset(references, sky, camera_altitude_deg=CAMERA_ALTITUDE)
    assert fit is not None
    assert fit.n_fixed == 0
    assert fit.dome_loss_mag is None
    assert fit.offset_mag == pytest.approx(OFFSET, abs=0.02)
    assert fit.altitude_mag_per_airmass == pytest.approx(K, abs=0.03)


def test_spoiled_readings_are_clipped() -> None:
    references, sky = synthetic_night(seed=3, outlier_fraction=0.08)
    fit = sqm_fit.fit_sqm_offset(references, sky, camera_altitude_deg=CAMERA_ALTITUDE)
    assert fit is not None
    assert fit.n_used < 160  # some pairs left the sample
    assert fit.n_used > 140
    assert fit.offset_mag == pytest.approx(OFFSET, abs=0.03)
    assert fit.dome_loss_mag == pytest.approx(DOME, abs=0.05)
    assert fit.rms_mag < 0.05


def test_an_offset_that_the_survey_already_applies_is_added_to_the_result() -> None:
    references, sky = synthetic_night()
    plain = sqm_fit.fit_sqm_offset(references, sky, camera_altitude_deg=CAMERA_ALTITUDE)
    shifted = sqm_fit.fit_sqm_offset(
        references, sky, camera_altitude_deg=CAMERA_ALTITUDE, applied_offset_mag=0.5
    )
    assert plain is not None
    assert shifted is not None
    assert shifted.offset_mag == pytest.approx(plain.offset_mag + 0.5)


def test_too_few_pairs_give_no_fit() -> None:
    references, sky = synthetic_night(n_fixed=6, manual_altitudes=(), n_manual_each=0)
    assert sqm_fit.fit_sqm_offset(references, sky, camera_altitude_deg=CAMERA_ALTITUDE) is None
    assert sqm_fit.fit_sqm_offset([], [], camera_altitude_deg=CAMERA_ALTITUDE) is None


def test_a_reading_pairs_with_the_nearest_sky_value_within_the_gap() -> None:
    sky = [
        sky_record(T0, 20.0),
        sky_record(T0 + 600 * NS_PER_S, 21.0),
        sky_record(T0 + 90 * NS_PER_S, None),
    ]
    refs = [
        reading(T0 + 100 * NS_PER_S, 18.0, source="fixed", altitude=45.0),  # nearest: the 20.0
        reading(T0 + 500 * NS_PER_S, 19.0, source="fixed", altitude=45.0),  # nearest: the 21.0
        reading(T0 + 5000 * NS_PER_S, 19.0, source="fixed", altitude=45.0),  # too far from both
        reading(T0 + 200 * NS_PER_S, 18.5, source="manual", altitude=None),  # no altitude
    ]
    pairs = sqm_fit.pair_readings(refs, sky)
    assert [(p.reading_mag, p.ours_mag) for p in pairs] == [(18.0, 20.0), (19.0, 21.0)]
    with_default = sqm_fit.pair_readings(refs, sky, default_altitude_deg=50.0)
    assert len(with_default) == 3
    assert with_default[-1].altitude_deg == 50.0
    assert sqm_fit.pair_readings(refs, [], default_altitude_deg=50.0) == []
    wide = sqm_fit.pair_readings(refs, sky, max_gap_s=6000.0)
    assert len(wide) == 3  # the 5000 s reading joins


def test_a_camera_at_the_altitude_of_the_meter_has_no_altitude_term() -> None:
    references, sky = synthetic_night(
        manual_altitudes=(), n_fixed=150, camera_altitude=SQM_ALTITUDE
    )
    fit = sqm_fit.fit_sqm_offset(references, sky, camera_altitude_deg=SQM_ALTITUDE)
    assert fit is not None
    # With the camera at 45 degrees, the airmass difference is zero, and offset + dome remain.
    assert fit.offset_mag == pytest.approx(OFFSET + DOME, abs=0.02)
