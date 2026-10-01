"""Transparency against a reference, the history protocol, clouds, and the limiting magnitude."""

from __future__ import annotations

import math

import numpy as np
import pytest

from seeingmon.clock import NS_PER_S, iso_to_utc_ns
from seeingmon.records.survey import SkyQualityRecord
from seeingmon.survey import transparency as tr
from seeingmon.survey.nights import night_label, night_start_utc_ns

DAY_NS = 86_400 * NS_PER_S
NOW_NS = iso_to_utc_ns("2026-10-01T22:00:00Z")


def sample(
    days_ago: float,
    zero_point: float = 19.2,
    *,
    rms: float | None = 0.02,
    n_stars: int = 80,
    cloud: float | None = 0.0,
) -> tr.ZeroPointSample:
    return tr.ZeroPointSample(
        t_utc_ns=NOW_NS - round(days_ago * DAY_NS),
        zero_point_mag=zero_point,
        rms_mag=rms,
        n_stars=n_stars,
        cloud_fraction=cloud,
    )


def record(t_utc_ns: int, zero_point: float | None = 19.2, **extra: object) -> SkyQualityRecord:
    return SkyQualityRecord(
        station_id="test-station",
        t_utc_ns=t_utc_ns,
        profile_id="profile-1",
        provenance={"algo": "sky-1"},
        zero_point_mag=zero_point,
        zero_point_rms_mag=0.02 if zero_point is not None else None,
        n_stars_used=80 if zero_point is not None else 0,
        cloud_fraction=0.05,
        **extra,  # type: ignore[arg-type]
    )


# --- Nights --------------------------------------------------------------------------------


def test_a_night_runs_from_noon_to_noon_utc() -> None:
    evening = iso_to_utc_ns("2026-10-01T18:30:00Z")
    after_midnight = iso_to_utc_ns("2026-10-02T03:00:00Z")
    next_day = iso_to_utc_ns("2026-10-02T12:00:00Z")
    assert night_label(evening) == "2026-10-01"
    assert night_label(after_midnight) == "2026-10-01"
    assert night_label(next_day) == "2026-10-02"
    assert night_label(iso_to_utc_ns("2026-10-02T11:59:59Z")) == "2026-10-01"
    # Another site chooses another hour.
    assert night_label(after_midnight, split_utc_hour=4.0) == "2026-10-01"
    assert night_label(after_midnight, split_utc_hour=2.0) == "2026-10-02"
    assert night_label(iso_to_utc_ns("2026-10-02T01:00:00Z"), split_utc_hour=2.0) == "2026-10-01"


def test_the_start_of_a_night_is_the_split_hour_of_its_date() -> None:
    assert night_start_utc_ns("2026-10-01") == iso_to_utc_ns("2026-10-01T12:00:00Z")
    assert night_start_utc_ns("2026-10-01", 14.5) == iso_to_utc_ns("2026-10-01T14:30:00Z")
    start = night_start_utc_ns("2026-03-30", 9.0)
    assert night_label(start, 9.0) == "2026-03-30"
    assert night_label(start - 1, 9.0) == "2026-03-29"


# --- The history ---------------------------------------------------------------------------


def test_the_memory_history_answers_time_ranges_in_order() -> None:
    history = tr.MemoryHistory()
    for days in (5.0, 1.0, 3.0, 2.0, 4.0):  # added out of order
        history.add(sample(days, 19.0 + days))
    assert len(history) == 5
    everything = history.zero_points(0, NOW_NS + 1)
    assert [s.t_utc_ns for s in everything] == sorted(s.t_utc_ns for s in everything)
    window = history.zero_points(NOW_NS - 3 * DAY_NS, NOW_NS - 1 * DAY_NS)
    assert [round((NOW_NS - s.t_utc_ns) / DAY_NS) for s in window] == [3, 2]  # the end is open
    assert history.zero_points(NOW_NS + 1, NOW_NS + 2) == ()


def test_the_memory_history_drops_the_oldest_beyond_its_limit() -> None:
    history = tr.MemoryHistory(max_samples=3)
    for days in (10.0, 9.0, 8.0, 7.0, 6.0):
        history.add(sample(days))
    kept = history.zero_points(0, NOW_NS + 1)
    assert [round((NOW_NS - s.t_utc_ns) / DAY_NS) for s in kept] == [8, 7, 6]
    with pytest.raises(ValueError, match="at least 1"):
        tr.MemoryHistory(max_samples=0)


def test_a_sky_quality_record_gives_a_sample_when_it_has_a_zero_point() -> None:
    with_zp = record(NOW_NS, 19.31)
    without = record(NOW_NS, None)
    sample_ = tr.sample_from_record(with_zp)
    assert sample_ == tr.ZeroPointSample(NOW_NS, 19.31, 0.02, 80, 0.05)
    assert tr.sample_from_record(without) is None
    assert len(tr.samples_from_records([with_zp, without, record(NOW_NS + 1, 19.2)])) == 2
    history = tr.MemoryHistory()
    assert history.add_record(with_zp)
    assert not history.add_record(without)
    assert len(history) == 1


def test_a_store_can_stand_behind_the_protocol() -> None:
    """Any object with `zero_points(since, until)` serves as a history."""

    class FakeStore:
        def zero_points(self, since_utc_ns: int, until_utc_ns: int) -> list[tr.ZeroPointSample]:
            return [sample(d) for d in range(1, 30)]

    reference = tr.reference_zero_point(FakeStore(), NOW_NS)
    assert reference is not None
    assert reference.n_samples == 29


# --- The reference and the transparency ---------------------------------------------------


def test_the_reference_is_a_high_quantile_of_the_clear_zero_points() -> None:
    values = np.linspace(18.9, 19.3, 41)
    history = tr.MemoryHistory(sample(1.0 + i * 0.5, float(v)) for i, v in enumerate(values))
    reference = tr.reference_zero_point(history, NOW_NS)
    assert reference is not None
    assert reference.zero_point_mag == pytest.approx(np.quantile(values, 0.9))
    assert reference.n_samples == 41
    assert reference.n_nights == 21  # two samples fall on one night (12 hours each side of noon)
    assert reference.quantile == 0.9
    assert reference.window_days == 60.0


def test_a_short_history_gives_no_reference() -> None:
    history = tr.MemoryHistory(sample(1.0 + i) for i in range(19))
    assert tr.reference_zero_point(history, NOW_NS) is None
    history.add(sample(25.0))
    assert tr.reference_zero_point(history, NOW_NS) is not None


def test_only_the_window_counts() -> None:
    old = [sample(70.0 + i, 20.0) for i in range(30)]  # a better zero point, but too old
    recent = [sample(1.0 + i, 19.0) for i in range(25)]
    reference = tr.reference_zero_point(tr.MemoryHistory(old + recent), NOW_NS)
    assert reference is not None
    assert reference.zero_point_mag == pytest.approx(19.0)
    longer = tr.TransparencyOptions(window_days=120.0)
    wide = tr.reference_zero_point(tr.MemoryHistory(old + recent), NOW_NS, longer)
    assert wide is not None
    assert wide.zero_point_mag > 19.0  # the old, better zero points count now


def test_cloudy_thin_and_scattered_zero_points_do_not_set_the_reference() -> None:
    clear = [sample(1.0 + i, 19.0) for i in range(25)]
    bad = (
        [sample(0.2 + i / 10, 19.8, cloud=0.5) for i in range(10)]  # clouds
        + [sample(0.3 + i / 10, 19.8, n_stars=5) for i in range(10)]  # few stars
        + [sample(0.4 + i / 10, 19.8, rms=0.4) for i in range(10)]  # a poor fit
        + [sample(0.5 + i / 10, float("nan")) for i in range(3)]
    )
    reference = tr.reference_zero_point(tr.MemoryHistory(clear + bad), NOW_NS)
    assert reference is not None
    assert reference.zero_point_mag == pytest.approx(19.0)
    assert reference.n_samples == 25
    # A sample with no scatter or cloud information counts.
    unknown = [sample(1.0 + i, 19.0, rms=None, cloud=None) for i in range(25)]
    assert tr.reference_zero_point(tr.MemoryHistory(unknown), NOW_NS) is not None


def test_the_reference_survives_a_few_lucky_frames() -> None:
    rng = np.random.default_rng(1)
    typical = [sample(i / 5.0 + 0.1, 19.0 + rng.normal(0.0, 0.03)) for i in range(200)]
    lucky = [sample(0.05 + i / 100.0, 19.6) for i in range(5)]
    reference = tr.reference_zero_point(tr.MemoryHistory(typical + lucky), NOW_NS)
    assert reference is not None
    assert reference.zero_point_mag < 19.1  # the 90th percentile of 205 frames ignores 5 outliers


def test_transparency_follows_the_zero_point_offset() -> None:
    reference = tr.ZeroPointReference(19.2, 100, 30, 60.0, 0.9)
    assert tr.transparency(19.2, reference) == pytest.approx(1.0)
    assert tr.transparency(19.2 - 0.75, reference) == pytest.approx(10 ** (-0.4 * 0.75))
    assert tr.transparency(19.2 - 0.75, reference) == pytest.approx(0.5012, abs=1e-3)
    assert tr.transparency(19.25, reference) > 1.0  # a frame clearer than the reference
    assert tr.transparency(19.2 - 2.5, reference) == pytest.approx(0.1)


def test_the_cloud_flag_follows_the_fraction_or_the_transparency() -> None:
    options = tr.TransparencyOptions()
    assert tr.cloud_flag(0.31, 0.9, options)
    assert tr.cloud_flag(0.05, 0.5, options)
    assert not tr.cloud_flag(0.05, 0.9, options)
    assert not tr.cloud_flag(None, None, options)
    assert tr.cloud_flag(None, 0.4, options)
    assert tr.cloud_flag(0.3, None, options)


def test_the_options_refuse_nonsense() -> None:
    with pytest.raises(ValueError, match="invalid transparency options"):
        tr.TransparencyOptions(quantile=0.0)
    with pytest.raises(ValueError, match="invalid transparency options"):
        tr.TransparencyOptions(window_days=-1.0)


# --- Detectability and the limiting magnitude ---------------------------------------------


def test_the_minimum_flux_gives_the_requested_signal_to_noise() -> None:
    for snr, noise, area in [(5.0, 11.5, 40.0), (20.0, 11.5, 78.0), (5.0, 2.0, 12.0)]:
        flux = tr.min_detectable_flux_e(snr, noise, area)
        assert flux / math.sqrt(flux + area * noise**2) == pytest.approx(snr, rel=1e-9)


def logistic_catalog(
    g50: float, *, n: int = 1500, width: float = 0.35, seed: int = 0, g_max: float = 13.0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = 9.0 + (g_max - 9.0) * rng.random(n) ** 0.6  # more faint stars than bright ones
    p = 1.0 / (1.0 + np.exp((g - g50) / width))
    return g, rng.random(n) < p


@pytest.mark.parametrize("g50", [10.4, 11.3, 12.1])
def test_the_limit_is_where_half_of_the_stars_are_detected(g50: float) -> None:
    g, found = logistic_catalog(g50, seed=int(g50 * 10))
    result = tr.limiting_magnitude(g, found)
    assert result.status == "measured"
    assert result.value == pytest.approx(g50, abs=0.2)
    assert result.n_stars == 1500


def test_a_clear_sky_has_no_measured_limit_and_the_noise_predicts_one() -> None:
    g, found = logistic_catalog(16.0, seed=1)  # the half point lies far beyond the catalog
    result = tr.limiting_magnitude(g, found)
    assert result.value is None
    assert result.status == "beyond_catalog"
    predicted = tr.limiting_magnitude(g, found, predicted_mag=16.4)
    assert predicted.value == 16.4
    assert predicted.status == "predicted"


def test_a_frame_without_detections_and_a_frame_with_few_stars_have_no_limit() -> None:
    g, _ = logistic_catalog(11.0, seed=2)
    nothing = tr.limiting_magnitude(g, np.zeros(g.size, dtype=bool))
    assert (nothing.value, nothing.status) == (None, "no_detections")
    few = tr.limiting_magnitude(g[:20], np.ones(20, dtype=bool))
    assert (few.value, few.status) == (None, "too_few_stars")
    empty = tr.limiting_magnitude(np.zeros(0), np.zeros(0, dtype=bool))
    assert empty.status == "too_few_stars"


def test_noisy_fractions_are_made_to_fall_steadily() -> None:
    fraction = np.array([1.0, 0.9, 0.95, 0.6, 0.7, 0.3, 0.1])
    weight = np.ones(7)
    smooth = tr._decreasing(fraction, weight)
    assert np.all(np.diff(smooth) <= 1e-12)
    assert smooth[0] == 1.0
    assert smooth.mean() == pytest.approx(fraction.mean())  # pooling keeps the total
    assert list(tr._decreasing(np.array([0.9, 0.5, 0.1]), np.ones(3))) == [0.9, 0.5, 0.1]


def test_clouds_move_the_limit_to_brighter_stars() -> None:
    clear_g, clear_found = logistic_catalog(12.2, seed=3)
    cloudy_g, cloudy_found = logistic_catalog(10.4, seed=4)
    clear = tr.limiting_magnitude(clear_g, clear_found)
    cloudy = tr.limiting_magnitude(cloudy_g, cloudy_found)
    assert clear.value is not None
    assert cloudy.value is not None
    assert cloudy.value < clear.value - 1.2
