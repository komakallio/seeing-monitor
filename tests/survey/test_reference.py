"""The pointing reference: the choice of a record, and the rebuild of its solution."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from seeingmon.store.db import Store, StoreReader
from seeingmon.survey import pointing as pt
from seeingmon.survey import reference as ref
from tests.survey import synth
from tests.survey.pointfx import (
    BIN2,
    CENTER,
    HOUR_NS,
    MINUTE_NS,
    PROFILE,
    SCALE_RAD,
    T0,
    Made,
    made,
    mount,
)

NOW = T0 + 10 * MINUTE_NS


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "results.sqlite") as opened:
        yield opened


def find(
    store: Store, *, min_matched: int = 100, max_age_s: float | None = 3600.0, now: int = NOW
) -> int:
    """The time of the record that the search finds."""
    with StoreReader.open(store.path) as reader:
        record = ref.newest_solution_record(
            reader, now_utc_ns=now, min_matched=min_matched, max_age_s=max_age_s
        )
    return record.t_utc_ns


def write(store: Store, *items: Made) -> None:
    for item in items:
        store.write(item.record)


# --- Which record -------------------------------------------------------------------------------


def test_the_newest_record_that_fits_wins(store: Store) -> None:
    write(
        store,
        made(T0),
        made(T0 + 2 * MINUTE_NS),
        made(T0 + 4 * MINUTE_NS),  # the newest good record
        made(T0 + 5 * MINUTE_NS, n_matched=99),
        made(T0 + 6 * MINUTE_NS, solved=False),
        made(T0 + 7 * MINUTE_NS, time_invalid=True),
        made(T0 + 8 * MINUTE_NS, rms_arcsec=None),
    )
    assert find(store) == T0 + 4 * MINUTE_NS


def test_the_age_limit_counts_from_now_and_none_sets_no_limit(store: Store) -> None:
    write(store, made(T0))
    assert find(store, now=T0 + 3600 * 1_000_000_000) == T0  # exactly at the limit
    with pytest.raises(ref.NoSolutionError, match="the newest pointing record is 60 min old"):
        find(store, now=T0 + 3601 * 1_000_000_000)
    assert find(store, now=T0 + 400 * HOUR_NS, max_age_s=None) == T0


def test_a_record_from_the_future_counts(store: Store) -> None:
    write(store, made(T0 + HOUR_NS))  # a clock that runs ahead of this machine
    assert find(store) == T0 + HOUR_NS


def test_a_huge_limit_does_not_overflow(store: Store) -> None:
    write(store, made(T0))
    assert find(store, max_age_s=1e300) == T0


@pytest.mark.parametrize(
    ("kinds", "reason"),
    [
        ([], "the store holds no pointing record"),
        (["unsolved"], "no pointing record of the last 60 minutes has a solution"),
        (
            ["unsolved", "unsolved", "untimed"],
            "every pointing solution of the last 60 minutes has the time_invalid flag",
        ),
        (
            ["unsolved", "untimed", "thin"],
            "no pointing solution of the last 60 minutes has 100 matched stars: "
            "the best has 30 (see --min-matched)",
        ),
        (
            ["unsolved", "untimed", "thin", "unmeasured"],
            "no pointing solution of the last 60 minutes has a finite residual",
        ),
    ],
)
def test_the_reason_names_the_check_that_the_best_record_passed_furthest(
    store: Store, kinds: list[str], reason: str
) -> None:
    items = {
        "unsolved": lambda t: made(t, solved=False),
        "untimed": lambda t: made(t, time_invalid=True),
        "thin": lambda t: made(t, n_matched=30),
        "unmeasured": lambda t: made(t, rms_arcsec=None),
    }
    write(store, *(items[kind](T0 + n * MINUTE_NS) for n, kind in enumerate(kinds)))
    with pytest.raises(ref.NoSolutionError) as raised:
        find(store)
    assert str(raised.value) == reason


def test_the_window_in_the_reason_names_the_limit(store: Store) -> None:
    write(store, made(T0, solved=False))
    with pytest.raises(ref.NoSolutionError) as raised:
        find(store, max_age_s=90.0, now=T0 + MINUTE_NS)
    assert str(raised.value) == "no pointing record of the last 1.5 minutes has a solution"
    with pytest.raises(ref.NoSolutionError) as raised:
        find(store, max_age_s=60.0, now=T0 + MINUTE_NS)
    assert str(raised.value) == "no pointing record of the last 1 minute has a solution"
    with pytest.raises(ref.NoSolutionError) as raised:
        find(store, max_age_s=None)
    assert str(raised.value) == "no pointing record of the store has a solution"


def test_a_record_that_a_writer_adds_between_two_searches_counts_in_the_second(
    store: Store,
) -> None:
    write(store, made(T0))
    with StoreReader.open(store.path) as reader:
        before = ref.newest_solution_record(
            reader, now_utc_ns=NOW, min_matched=100, max_age_s=3600.0
        )
        write(store, made(T0 + MINUTE_NS))  # a record that arrives after the first search
        after = ref.newest_solution_record(
            reader, now_utc_ns=NOW, min_matched=100, max_age_s=3600.0
        )
    assert (before.t_utc_ns, after.t_utc_ns) == (T0, T0 + MINUTE_NS)


# --- The solution -------------------------------------------------------------------------------


def test_the_rebuilt_solution_is_the_one_that_the_fit_made() -> None:
    item = made(T0, n_matched=321, rms_arcsec=0.7, solver="astap")
    solution = ref.solution_from_record(item.record, PROFILE)
    original = item.solution
    np.testing.assert_allclose(
        solution.rotation_earth_fixed, original.rotation_earth_fixed, atol=1e-13
    )
    assert solution.scale_rad_px == pytest.approx(SCALE_RAD, rel=1e-13)
    assert (solution.parity, solution.mode) == (1, "bin2")
    assert (solution.width_px, solution.height_px, solution.center_px) == (
        BIN2.width_px,
        BIN2.height_px,
        CENTER,
    )
    assert solution.t_utc_ns == T0
    assert (solution.n_matched, solution.rms_arcsec, solution.solver) == (321, 0.7, "astap")
    assert pt.offset_between(solution, original).boresight_arcmin == pytest.approx(0.0, abs=1e-9)


def test_a_json_round_trip_keeps_the_solution(tmp_path: Path) -> None:
    item = made(T0)
    reference = pt.ReferenceSolution("x", ref.solution_from_record(item.record, PROFILE))
    pt.save_reference(tmp_path / "ref.json", reference)
    loaded = pt.load_reference(tmp_path / "ref.json")
    assert loaded is not None
    np.testing.assert_allclose(
        loaded.solution.rotation_earth_fixed, item.solution.rotation_earth_fixed, atol=1e-13
    )


def test_a_boresight_far_from_the_pole_still_gives_the_parity() -> None:
    # Polaris sits 0.66 degrees from the pole, so a camera that points 1.5 degrees off puts it far
    # from the middle of the frame, and perhaps outside the frame. The record holds where Polaris
    # falls all the same, and that tells the parity.
    for parity in (1, -1):
        item = made(T0, rotation_tirs=mount(1.5), parity=parity)
        assert ref.solution_from_record(item.record, PROFILE).parity == parity


def test_a_record_that_cannot_tell_the_parity_gets_the_plain_image() -> None:
    item = made(T0, parity=-1)
    blind = item.record.model_copy(
        update={"polaris_x_px": None, "polaris_y_px": None, "roll_deg": None}
    )
    assert ref.solution_from_record(blind, PROFILE).parity == 1
    only_roll = item.record.model_copy(update={"polaris_x_px": None, "polaris_y_px": None})
    assert ref.solution_from_record(only_roll, PROFILE).parity == -1  # the roll tells it


def test_a_record_without_a_solution_cannot_be_rebuilt() -> None:
    with pytest.raises(ref.PointingReferenceError, match="has no solution"):
        ref.solution_from_record(made(T0, solved=False).record, PROFILE)


def test_a_mode_that_the_profile_lacks_is_refused() -> None:
    item = made(T0)
    odd = item.record.model_copy(update={"readout_mode": "bin9"})
    with pytest.raises(
        ref.PointingReferenceError, match="does not fit the profile: unknown readout mode"
    ):
        ref.solution_from_record(odd, PROFILE)


@pytest.mark.parametrize(
    "attitude",
    [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],  # a scale
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],  # a mirror
        [float("nan")] * 9,
    ],
)
def test_an_attitude_that_is_no_rotation_is_refused(attitude: list[float]) -> None:
    item = made(T0)
    broken = item.record.model_copy(update={"attitude": attitude})
    with pytest.raises(ref.PointingReferenceError, match="is not a rotation"):
        ref.solution_from_record(broken, PROFILE)


def test_a_record_of_another_profile_is_refused() -> None:
    item = made(T0)
    with pytest.raises(ref.PointingReferenceError, match="puts Polaris away from the position"):
        ref.solution_from_record(item.record, synth.cropped_profile(1200, 800))


def test_a_polaris_that_is_a_pixel_off_is_refused() -> None:
    item = made(T0)
    assert item.record.polaris_x_px is not None
    nudged = item.record.model_copy(update={"polaris_x_px": item.record.polaris_x_px + 1.0})
    with pytest.raises(ref.PointingReferenceError, match="puts Polaris away"):
        ref.solution_from_record(nudged, PROFILE)
    near = item.record.model_copy(update={"polaris_x_px": item.record.polaris_x_px + 0.005})
    assert ref.solution_from_record(near, PROFILE).parity == 1  # within 0.01 pixel


def test_a_wrong_roll_is_refused() -> None:
    item = made(T0)
    assert item.record.roll_deg is not None
    turned = item.record.model_copy(update={"roll_deg": item.record.roll_deg + 1.0})
    with pytest.raises(ref.PointingReferenceError, match="puts Polaris away"):
        ref.solution_from_record(turned, PROFILE)


def test_dut1_moves_the_earth_fixed_attitude_about_the_pole_only() -> None:
    item = made(T0)
    plain = ref.solution_from_record(item.record, PROFILE)
    shifted = ref.solution_from_record(item.record, PROFILE, dut1_s=0.9)
    assert shifted.dut1_s == 0.9
    offset = pt.offset_between(shifted, plain)
    # 0.9 s of UT1 - UTC is 13.5 arcsec of Earth rotation. A camera 0.4 degrees from the pole
    # moves by that angle times the sine of the angle from the pole.
    expected_arcsec = 15.0411 * 0.9 * np.sin(np.radians(0.4))
    assert offset.boresight_arcmin * 60.0 == pytest.approx(expected_arcsec, rel=2e-3)


# --- The text -----------------------------------------------------------------------------------


def test_the_default_id_names_the_utc_time_of_the_solution() -> None:
    assert ref.default_reference_id(T0) == "reference-20261001T220000Z"
    assert (
        ref.default_reference_id(T0 + 7_000_000_000 + 999_999_999) == "reference-20261001T220007Z"
    )
    assert ref.is_valid_reference_id(ref.default_reference_id(T0))


@pytest.mark.parametrize("text", ["a", "7", "commissioning-1", "A.b_c-d", "x" * 64])
def test_these_ids_are_valid(text: str) -> None:
    assert ref.is_valid_reference_id(text)


@pytest.mark.parametrize(
    "text",
    ["", " ", "has space", "-dash", ".dot", "_under", "x" * 65, "tab\t", "new\nline", "end\n", "ä"],
)
def test_these_ids_are_not(text: str) -> None:
    assert not ref.is_valid_reference_id(text)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (-5.0, "0 s"),
        (45.0, "45 s"),
        (89.0, "89 s"),
        (600.0, "10 min"),
        (89 * 60.0, "89 min"),
        (3 * 3600.0, "3.0 h"),
        (47 * 3600.0, "47.0 h"),
        (3 * 86400.0, "3.0 days"),
    ],
)
def test_the_age_is_short(seconds: float, text: str) -> None:
    assert ref.format_age(seconds) == text


@pytest.mark.parametrize(
    ("value", "digits", "text"),
    [
        (1.236, 2, "1.24"),
        (-0.004, 2, "0.00"),
        (-0.0, 2, "0.00"),
        (-0.006, 2, "-0.01"),
        (12.0, 0, "12"),
    ],
)
def test_a_number_that_rounds_to_zero_has_no_sign(value: float, digits: int, text: str) -> None:
    assert ref.format_number(value, digits) == text


def test_the_summary_rows() -> None:
    item = made(T0, rotation_tirs=mount(0.25), rms_arcsec=0.456)
    reference = pt.ReferenceSolution(
        "commissioning-1", ref.solution_from_record(item.record, PROFILE)
    )
    rows = dict(ref.summary_rows(reference, now_utc_ns=T0 + 125 * MINUTE_NS))
    assert rows == {
        "reference ID": "commissioning-1",
        "solution time": "2026-10-01T22:00:00Z (2.1 h ago)",
        "matched stars": "300",
        "residual": "0.46 arcsec rms",
        "roll": "25.00 degrees",
        "plate scale": "3.820 arcsec/px in bin2",
        "center from the pole": "0.250 degrees",
    }
    plain = dict(ref.summary_rows(reference))
    assert plain["solution time"] == "2026-10-01T22:00:00Z"
    blank = pt.ReferenceSolution(
        "x",
        ref.solution_from_record(
            made(T0, rotation_tirs=mount(0.0), rms_arcsec=None).record, PROFILE
        ),
    )
    rows = dict(ref.summary_rows(blank))
    assert rows["residual"] == "not recorded"
    assert rows["roll"] == "undefined (the pole is at the center)"
    assert rows["center from the pole"] == "0.000 degrees"


def test_the_angle_from_the_pole_is_exact_for_a_camera_on_the_pole_and_for_one_beside_it() -> None:
    for degrees in (0.0, 0.01, 0.5, 1.3):
        solution = ref.solution_from_record(made(T0, rotation_tirs=mount(degrees)).record, PROFILE)
        assert ref.angle_from_pole_deg(solution) == pytest.approx(degrees, abs=1e-9)
