"""The pointing tracker: Polaris predictions, the age limit, modes, and predict-match-fit."""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

from seeingmon.analysis import PointingProvider
from seeingmon.clock import NS_PER_S
from seeingmon.survey import apparent
from seeingmon.survey import pointing as pt
from seeingmon.survey.catalog import CapCatalog
from seeingmon.survey.geometry import ARCSEC_PER_RAD, exp_so3
from seeingmon.survey.tracker import PointingTracker
from seeingmon.survey.wcs_fit import CameraAttitude, FitOptions
from tests.survey import synth

T0 = synth.NIGHT_UTC_NS
DAY_NS = 86_400 * NS_PER_S
PROFILE = synth.reference_profile()
SHAPE = (2822, 4144)


def truth_at(
    t_utc_ns: int, *, catalog: CapCatalog | None = None, polar_distance: float = 0.9
) -> tuple[synth.SynthTruth, CapCatalog]:
    """The geometry of a frame of the reference camera at a time (a mount that does not move)."""
    star_catalog = (
        catalog if catalog is not None else synth.synthetic_catalog(cap_radius_deg=6.0, seed=21)
    )
    truth, _, _ = synth.star_truth(
        star_catalog,
        PROFILE,
        rotation_tirs=synth.make_attitude(polar_distance, 40.0, 25.0),
        t_utc_ns=t_utc_ns,
        exposure_s=0.001,
    )
    return truth, star_catalog


def solution_from_truth(truth: synth.SynthTruth, *, shift_px: float = 0.0) -> pt.PointingSolution:
    attitude = CameraAttitude(
        truth.rotation_cirs, truth.scale_arcsec_px / ARCSEC_PER_RAD, truth.parity, truth.center_px
    )
    if shift_px:
        delta = np.array([shift_px, -0.6 * shift_px, 0.0]) * attitude.scale_rad_px
        attitude = attitude.with_rotation(exp_so3(delta) @ attitude.rotation)
    return pt.PointingSolution.from_attitude(
        attitude, truth.epoch, mode="bin2", width_px=4144, height_px=2822, n_matched=100,
        solver="test",
    )  # fmt: skip


def holds_untimed(tracker: PointingTracker) -> bool:
    """Whether the tracker holds an untimed solution (a function, so mypy does not narrow it)."""
    return tracker.untimed


def test_the_tracker_is_a_pointing_provider() -> None:
    assert isinstance(PointingTracker(PROFILE), PointingProvider)


def test_without_a_solution_there_is_no_polaris_position() -> None:
    tracker = PointingTracker(PROFILE)
    assert tracker.polaris_position(T0, "bin2") is None
    assert tracker.solution is None
    assert tracker.age_s(T0) is None
    assert not tracker.valid_at(T0)
    assert tracker.attitude_at(T0) is None
    assert tracker.trail_model(T0, 30.0) is None


def test_the_polaris_prediction_follows_the_true_position_as_the_sky_turns() -> None:
    truth0, catalog = truth_at(T0)
    tracker = PointingTracker(PROFILE)
    assert tracker.update(solution_from_truth(truth0))
    polaris_row = int(np.argmin(catalog.g_mag))
    for seconds in (0, 60, 600, 3 * 3600, -1800):
        t = T0 + seconds * NS_PER_S
        truth, _ = truth_at(t, catalog=catalog)
        where = np.flatnonzero(truth.rows == polaris_row)
        assert where.size == 1
        expected = (float(truth.x[where[0]]), float(truth.y[where[0]]))
        predicted = tracker.polaris_position(t, "bin2")
        assert predicted is not None
        # The catalog moves Polaris with the same constants, so the prediction is exact up to
        # the first-order proper motion.
        assert predicted == pytest.approx(expected, abs=2e-3)


def test_the_prediction_converts_to_another_readout_mode() -> None:
    truth, _ = truth_at(T0)
    tracker = PointingTracker(PROFILE)
    tracker.update(solution_from_truth(truth))
    bin2 = tracker.polaris_position(T0, "bin2")
    bin1 = tracker.polaris_position(T0, "bin1")
    assert bin2 is not None
    assert bin1 is not None
    # The center of bin2 pixel 0 lies halfway between the centers of bin1 pixels 0 and 1.
    assert bin1 == pytest.approx(((bin2[0] + 0.5) * 2 - 0.5, (bin2[1] + 0.5) * 2 - 0.5))
    assert tracker.convert(0.0, 0.0, "bin2", "bin1") == (0.5, 0.5)
    assert tracker.convert(10.0, 20.0, "bin1", "bin1") == (10.0, 20.0)
    with pytest.raises(Exception, match="unknown readout mode"):
        tracker.polaris_position(T0, "nonexistent")


def test_a_positive_validity_limit_still_expires_the_solution() -> None:
    truth, _ = truth_at(T0)
    tracker = PointingTracker(PROFILE, validity_s=3600.0)
    tracker.update(solution_from_truth(truth))
    assert tracker.polaris_position(T0 + 3599 * NS_PER_S, "bin2") is not None
    assert tracker.polaris_position(T0 + 3601 * NS_PER_S, "bin2") is None
    assert tracker.polaris_position(T0 - 3601 * NS_PER_S, "bin2") is None
    assert tracker.polaris_position(T0 - 3000 * NS_PER_S, "bin2") is not None
    assert not tracker.valid_at(T0 + 3601 * NS_PER_S)
    assert tracker.attitude_at(T0 + 3601 * NS_PER_S) is None
    assert tracker.age_s(T0 + 120 * NS_PER_S) == 120.0
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="validity"):
            PointingTracker(PROFILE, validity_s=bad)


def test_a_solution_that_is_30_days_old_still_predicts_polaris() -> None:
    """The default has no age limit: a rigid mount keeps its Earth-fixed attitude."""
    truth0, catalog = truth_at(T0)
    tracker = PointingTracker(PROFILE)  # the default, validity_s 0, is no limit
    assert tracker.update(solution_from_truth(truth0))
    polaris_row = int(np.argmin(catalog.g_mag))
    for days in (30, -30):
        t = T0 + days * DAY_NS
        truth, _ = truth_at(t, catalog=catalog)
        (where,) = np.flatnonzero(truth.rows == polaris_row)
        predicted = tracker.polaris_position(t, "bin2")
        assert predicted is not None
        # The catalog moves Polaris with the same constants, so the error stays at the first-order
        # proper motion over a month: 2e-3 pixel, as for an age of hours.
        assert predicted == pytest.approx((float(truth.x[where]), float(truth.y[where])), abs=2e-3)
        assert tracker.valid_at(t)
        assert tracker.attitude_at(t) is not None
        assert tracker.trail_model(t, 30.0) is not None
    assert tracker.age_s(T0 + 30 * DAY_NS) == 30 * 86_400.0
    zero = PointingTracker(PROFILE, validity_s=0.0)  # 0 written out is the same as the default
    zero.update(solution_from_truth(truth0))
    assert zero.polaris_position(T0 + 30 * DAY_NS, "bin2") == tracker.polaris_position(
        T0 + 30 * DAY_NS, "bin2"
    )


def test_an_older_solution_does_not_replace_a_newer_one() -> None:
    truth_new, catalog = truth_at(T0 + 600 * NS_PER_S)
    truth_old, _ = truth_at(T0, catalog=catalog)
    tracker = PointingTracker(PROFILE)
    newer, older = solution_from_truth(truth_new), solution_from_truth(truth_old)
    assert tracker.update(newer)
    assert not tracker.update(older)
    held = tracker.solution
    assert held is newer
    tracker.clear()
    after_clear = tracker.solution
    assert after_clear is None
    assert tracker.update(older)


def test_a_solution_without_a_valid_time_holds_until_a_timed_one_replaces_it() -> None:
    """The solve of a frame that a clock stamped 6 hours ahead, as after a boot without time.

    The error of the clock cancels between the solve and the prediction, so the untimed solution
    places Polaris right at the times of the same clock. It lies in the future, so the order of
    time cannot judge it against a timed one.
    """
    ahead_ns = 6 * 3600 * NS_PER_S
    truth0, catalog = truth_at(T0)
    timed = solution_from_truth(truth0)
    attitude = CameraAttitude(
        truth0.rotation_cirs, truth0.scale_arcsec_px / ARCSEC_PER_RAD, truth0.parity,
        truth0.center_px,
    )  # fmt: skip
    untimed = pt.PointingSolution.from_attitude(
        attitude, apparent.epoch_from_utc_ns(T0 + ahead_ns), mode="bin2", width_px=4144,
        height_px=2822, n_matched=100, solver="test",
    )  # fmt: skip
    tracker = PointingTracker(PROFILE)
    assert tracker.update(untimed, timed=False)  # an empty tracker takes it
    assert holds_untimed(tracker)
    polaris_row = int(np.argmin(catalog.g_mag))
    truth, _ = truth_at(T0 + 600 * NS_PER_S, catalog=catalog)
    (where,) = np.flatnonzero(truth.rows == polaris_row)
    true_xy = (float(truth.x[where]), float(truth.y[where]))
    predicted = tracker.polaris_position(T0 + ahead_ns + 600 * NS_PER_S, "bin2")
    # The apparent places of the two epochs, 6 hours apart, differ by 0.02 pixel (0.08 arcsec).
    assert predicted == pytest.approx(true_xy, abs=0.05)  # pixels
    # After the clock steps back to the true time, the untimed solution points far off.
    stale = tracker.polaris_position(T0 + 600 * NS_PER_S, "bin2")
    assert stale is not None
    assert np.hypot(stale[0] - true_xy[0], stale[1] - true_xy[1]) > 100.0
    # Among untimed solutions, the order of time holds.
    assert not tracker.update(replace(untimed, t_utc_ns=untimed.t_utc_ns - NS_PER_S), timed=False)
    later = replace(untimed, t_utc_ns=untimed.t_utc_ns + NS_PER_S)
    assert tracker.update(later, timed=False)
    # A timed solution replaces it, although it is 6 hours older.
    assert tracker.update(timed)
    assert not holds_untimed(tracker)
    assert tracker.solution is timed
    # An untimed solution never replaces a timed one, whatever its time, and timed ones keep the
    # order of time.
    assert not tracker.update(later, timed=False)
    assert not tracker.update(replace(timed, t_utc_ns=T0 - NS_PER_S))
    assert tracker.solution is timed
    tracker.clear()  # a cleared tracker takes an untimed solution again
    assert not holds_untimed(tracker)
    assert tracker.update(later, timed=False)


def test_the_tracker_gives_the_trail_model_of_an_exposure() -> None:
    truth, _ = truth_at(T0)
    tracker = PointingTracker(PROFILE)
    tracker.update(solution_from_truth(truth))
    model = tracker.trail_model(T0, 30.0)
    assert model is not None
    assert (model.pole_x, model.pole_y) == pytest.approx(truth.pole_px, abs=1e-6)
    assert model.rotation_rad == pytest.approx(apparent.EARTH_ROTATION_RATE_RAD_S * 30.0)


def test_a_reference_gives_the_offset_of_a_new_solution() -> None:
    truth, _ = truth_at(T0)
    solution = solution_from_truth(truth)
    tracker = PointingTracker(PROFILE)
    assert tracker.offset_from_reference(solution) is None
    tracker.set_reference(pt.ReferenceSolution("ref", solution))
    assert tracker.reference is not None
    moved = solution_from_truth(truth, shift_px=10.0)
    offset = tracker.offset_from_reference(moved)
    assert offset is not None
    expected_arcmin = float(np.hypot(10.0, 6.0)) * 3.82 / 60.0
    assert offset.boresight_arcmin == pytest.approx(expected_arcmin, rel=0.01)


# --- Predict, match, fit ---------------------------------------------------------------------


def detections_of(
    truth: synth.SynthTruth, *, noise_px: float = 0.03, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inside = (truth.x > 5) & (truth.x < 4138) & (truth.y > 5) & (truth.y < 2816)
    count = int(inside.sum())
    return (
        truth.x[inside] + rng.normal(0, noise_px, count),
        truth.y[inside] + rng.normal(0, noise_px, count),
        np.full(count, noise_px),
    )


def test_track_recovers_the_pointing_after_the_mount_moved_a_few_pixels() -> None:
    truth_before, catalog = truth_at(T0)
    tracker = PointingTracker(PROFILE)
    tracker.update(solution_from_truth(truth_before, shift_px=3.5))  # the last solve was off
    t = T0 + 120 * NS_PER_S
    truth, _ = truth_at(t, catalog=catalog)
    x, y, error = detections_of(truth, seed=1)
    result = tracker.track(catalog, x, y, error, t_utc_ns=t, mode="bin2", shape=SHAPE)
    assert result is not None
    assert result.fit.n_matched > 100
    truth_model = CameraAttitude(
        truth.rotation_cirs, truth.scale_arcsec_px / ARCSEC_PER_RAD, 1, truth.center_px
    )
    gx, gy = np.meshgrid(np.linspace(0, 4143, 9), np.linspace(0, 2821, 7))
    vectors = truth_model.unproject(gx.ravel(), gy.ravel())
    px, py, _ = result.fit.attitude.project(vectors)
    worst = float(np.max(np.hypot(px - gx.ravel(), py - gy.ravel())))
    assert worst < 0.02  # pixels, over the whole field
    assert result.solution.solver == "tracker"
    assert result.solution.t_utc_ns == t
    assert result.rows.size > result.fit.n_matched
    assert result.vectors.shape == (result.rows.size, 3)


def test_track_commits_the_new_solution_on_request() -> None:
    truth, catalog = truth_at(T0)
    tracker = PointingTracker(PROFILE)
    tracker.update(solution_from_truth(truth, shift_px=2.0))
    t = T0 + 30 * NS_PER_S
    truth_later, _ = truth_at(t, catalog=catalog)
    x, y, error = detections_of(truth_later, seed=2)
    before = tracker.solution
    first = tracker.track(catalog, x, y, error, t_utc_ns=t, mode="bin2", shape=SHAPE)
    assert first is not None
    assert tracker.solution is before  # no commit by default
    second = tracker.track(
        catalog,
        x,
        y,
        error,
        t_utc_ns=t,
        mode="bin2",
        shape=SHAPE,
        commit=True,
    )
    assert second is not None
    assert tracker.solution is second.solution
    # With the new solution the Polaris prediction matches the truth closely.
    polaris_row = int(np.argmin(catalog.g_mag))
    where = int(np.flatnonzero(truth_later.rows == polaris_row)[0])
    predicted = tracker.polaris_position(t + 60 * NS_PER_S, "bin2")
    truth_next, _ = truth_at(t + 60 * NS_PER_S, catalog=catalog)
    where_next = int(np.flatnonzero(truth_next.rows == polaris_row)[0])
    assert predicted == pytest.approx(
        (float(truth_next.x[where_next]), float(truth_next.y[where_next])), abs=0.02
    )
    del where


def test_track_limits_the_magnitude_and_the_options() -> None:
    truth, catalog = truth_at(T0)
    tracker = PointingTracker(PROFILE)
    tracker.update(solution_from_truth(truth, shift_px=1.0))
    x, y, error = detections_of(truth, seed=3)
    options = FitOptions(match_radius_px=(3.0, 1.5))
    bright = tracker.track(
        catalog,
        x,
        y,
        error,
        t_utc_ns=T0,
        mode="bin2",
        shape=SHAPE,
        options=options,
        max_g_mag=10.5,
    )
    everything = tracker.track(
        catalog,
        x,
        y,
        error,
        t_utc_ns=T0,
        mode="bin2",
        shape=SHAPE,
        options=options,
    )
    assert bright is not None
    assert everything is not None
    assert bright.fit.n_matched < everything.fit.n_matched
    assert bright.rows.size < everything.rows.size


def test_track_returns_none_when_it_cannot_predict_or_match() -> None:
    truth, catalog = truth_at(T0)
    x, y, error = detections_of(truth, seed=4)
    tracker = PointingTracker(PROFILE, validity_s=600.0)

    def track(
        xs: np.ndarray, ys: np.ndarray, errors: np.ndarray, t: int = T0, mode: str = "bin2"
    ) -> object:
        return tracker.track(catalog, xs, ys, errors, t_utc_ns=t, mode=mode, shape=SHAPE)

    assert track(x, y, error) is None  # no solution yet
    tracker.update(solution_from_truth(truth))
    assert track(x, y, error, T0 + 700 * NS_PER_S) is None  # the solution has expired
    assert track(x, y, error, mode="bin1") is None  # the solution belongs to another mode
    assert track(x[:3], y[:3], error[:3]) is None  # too few detections
    empty = np.array([])
    assert track(empty, empty, empty) is None


def test_the_tracker_is_safe_to_use_from_two_threads() -> None:
    truth, _ = truth_at(T0)
    base = solution_from_truth(truth)
    tracker = PointingTracker(PROFILE)
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(200):
                tracker.update(replace(base, t_utc_ns=T0 + i * NS_PER_S))
        except BaseException as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(200):
                tracker.polaris_position(T0, "bin2")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
