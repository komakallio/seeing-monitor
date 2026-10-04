"""The alignment demo: the aim ring, the last solution, and a period without a solution."""

from __future__ import annotations

import json
import math

import pytest

from seeingmon.services.core.alignment.focus import FocusHistory
from seeingmon.services.web.contract import (
    MAX_STATE_BYTES,
    AlignmentState,
    decode_alignment_state,
    pack_frame,
)
from seeingmon.services.web.demo import (
    FOCUS_POINTS,
    FOCUS_SPIKE_EVERY,
    FRAME_CENTER_X,
    FRAME_CENTER_Y,
    FRAME_PERIOD_S,
    LOST_REASON,
    ORBIT_RADIUS_PX,
    PLATE_SCALE_ARCSEC_PX,
    POLE_DRIFT_PERIOD_S,
    SKY_COLATITUDE_DEG,
    SOLUTION_LOST_FROM_S,
    SOLUTION_LOST_UNTIL_S,
    DemoCore,
    StarField,
    demo_aim_ring,
    demo_fwhm_px,
    demo_roll_deg,
    demo_spike_flags,
    solution_lost,
)

FIRST_LOST = round(SOLUTION_LOST_FROM_S / FRAME_PERIOD_S)  # the first frame without a solution
FIRST_BACK = round(SOLUTION_LOST_UNTIL_S / FRAME_PERIOD_S)  # the first frame with one again


@pytest.fixture(scope="module")
def field() -> StarField:
    return StarField(stars=20)


def state_of(seq: int, field: StarField) -> AlignmentState:
    return field.frame(seq).state


def test_the_solver_loses_the_star_field_for_twenty_seconds_of_every_period() -> None:
    assert (SOLUTION_LOST_FROM_S, SOLUTION_LOST_UNTIL_S) == (50.0, 70.0)
    assert not solution_lost(49.5)
    assert solution_lost(50.0)
    assert solution_lost(69.5)
    assert not solution_lost(70.0)
    assert solution_lost(POLE_DRIFT_PERIOD_S + 55.0)  # it comes again in the next period
    assert not solution_lost(POLE_DRIFT_PERIOD_S + 20.0)


class TestWithASolution:
    def test_the_ring_comes_from_the_current_frame_and_agrees_with_the_sky(
        self, field: StarField
    ) -> None:
        for seq in (0, 20, FIRST_LOST - 1, FIRST_BACK, 150, 233):
            state = state_of(seq, field)
            assert state.solved is not None
            assert state.sky is not None
            assert state.sky.aim_ring is not None
            ring = state.aim_ring
            assert ring is not None
            assert ring.source == "current frame", seq
            assert (ring.x_px, ring.y_px) == pytest.approx(
                (state.sky.aim_ring.x_px, state.sky.aim_ring.y_px), abs=0.5
            )
            assert ring.age_s == pytest.approx(state.solved.age_s, abs=FRAME_PERIOD_S)
            assert state.quality == {}

    def test_the_ring_lies_on_the_circle_of_the_reticle(self, field: StarField) -> None:
        for seq in (0, 40, 150, 233, FIRST_LOST + 10, 400):
            state = state_of(seq, field)
            assert state.reticle is not None
            assert state.aim_ring is not None
            distance = math.hypot(
                state.aim_ring.x_px - state.reticle.x_px, state.aim_ring.y_px - state.reticle.y_px
            )
            assert distance == pytest.approx(state.reticle.radius_px, abs=0.01), seq

    def test_the_reticle_is_the_circle_of_the_orbit_around_the_frame_center(
        self, field: StarField
    ) -> None:
        reticle = state_of(3, field).reticle
        assert reticle is not None
        assert (reticle.x_px, reticle.y_px) == (FRAME_CENTER_X, FRAME_CENTER_Y)
        assert reticle.radius_px == pytest.approx(ORBIT_RADIUS_PX, abs=0.001)
        assert reticle.polaris_colatitude_deg == SKY_COLATITUDE_DEG

    def test_the_last_solution_is_a_fraction_of_a_second_old(self, field: StarField) -> None:
        state = state_of(20, field)
        assert state.last_solution is not None
        assert state.solved is not None
        assert state.last_solution.age_s == pytest.approx(state.solved.age_s, abs=0.01)
        assert state.last_solution.frame_seq <= 20
        assert state.timing is not None
        assert state.timing.solution_frame_seq == state.last_solution.frame_seq

    def test_the_zero_roll_puts_the_ring_where_the_target_is(self) -> None:
        field = StarField(stars=20)
        ring = demo_aim_ring(0.0)
        assert ring.x_px == pytest.approx(field.target_x)
        assert ring.y_px == pytest.approx(field.target_y, abs=0.01)


class TestWithoutASolution:
    def test_the_state_has_no_solution_no_offset_and_no_sky(self, field: StarField) -> None:
        for seq in (FIRST_LOST, FIRST_LOST + 15, FIRST_BACK - 1):
            state = state_of(seq, field)
            assert state.active is True
            assert state.solved is None
            assert state.offset is None
            assert state.sky is None  # the pole and the grid need the pointing of this frame
            assert state.quality["solved"] == LOST_REASON
            assert state.quality["sky"] == LOST_REASON
            assert "offset" in state.quality
            assert state.target is not None
            assert state.focus is not None  # the focus needs no solution
            assert state.reticle is not None  # the dashed circle stays

    def test_the_ring_comes_from_the_last_solution_and_does_not_move(
        self, field: StarField
    ) -> None:
        rings = []
        for seq in (FIRST_LOST, FIRST_LOST + 10, FIRST_BACK - 1):
            state = state_of(seq, field)
            assert state.aim_ring is not None
            assert state.aim_ring.source == "last solution"
            assert state.aim_ring.solution_frame_seq == FIRST_LOST - 1
            rings.append((state.aim_ring.x_px, state.aim_ring.y_px))
        assert len(set(rings)) == 1  # the roll of the last solution holds the ring still
        expected = demo_aim_ring(demo_roll_deg((FIRST_LOST - 1) * FRAME_PERIOD_S))
        assert rings[0] == (expected.x_px, expected.y_px)

    def test_the_last_solution_grows_old_during_the_period(self, field: StarField) -> None:
        ages = []
        for seq in (FIRST_LOST, FIRST_LOST + 10, FIRST_BACK - 1):
            state = state_of(seq, field)
            assert state.last_solution is not None
            assert state.aim_ring is not None
            assert state.last_solution.frame_seq == FIRST_LOST - 1
            assert state.aim_ring.age_s == state.last_solution.age_s
            ages.append(state.last_solution.age_s)
        assert ages == sorted(ages)
        assert ages[0] == pytest.approx(FRAME_PERIOD_S)
        assert ages[-1] == pytest.approx(SOLUTION_LOST_UNTIL_S - SOLUTION_LOST_FROM_S, abs=1.0)

    def test_the_solve_that_failed_is_the_latest_one(self, field: StarField) -> None:
        state = state_of(FIRST_LOST + 4, field)
        assert state.timing is not None
        assert (
            state.timing.solution_frame_seq == FIRST_LOST + 3
        )  # the failure of the previous frame
        assert state.timing.frame_seq == FIRST_LOST + 4

    def test_the_picture_goes_on_and_the_solution_comes_back(self, field: StarField) -> None:
        lost = field.frame(FIRST_LOST + 5)
        back = field.frame(FIRST_BACK)
        assert lost.jpeg.startswith(b"\xff\xd8")
        assert back.state.solved is not None
        assert back.state.aim_ring is not None
        assert back.state.aim_ring.source == "current frame"
        assert back.state.quality == {}

    def test_the_period_comes_again(self, field: StarField) -> None:
        seq = round((POLE_DRIFT_PERIOD_S + 55.0) / FRAME_PERIOD_S)
        state = state_of(seq, field)
        assert state.solved is None
        assert state.aim_ring is not None
        assert state.aim_ring.source == "last solution"
        assert state.last_solution is not None
        assert state.last_solution.frame_seq == round((POLE_DRIFT_PERIOD_S + 49.5) / FRAME_PERIOD_S)


class TestFocus:
    def test_the_value_comes_in_pixels_and_arcseconds(self, field: StarField) -> None:
        focus = state_of(60, field).focus
        assert focus is not None
        assert focus.fwhm_px is not None
        assert focus.fwhm_arcsec == pytest.approx(focus.fwhm_px * PLATE_SCALE_ARCSEC_PX, abs=1e-3)
        assert focus.best_fwhm_px is not None
        assert focus.best_fwhm_arcsec == pytest.approx(
            focus.best_fwhm_px * PLATE_SCALE_ARCSEC_PX, abs=1e-3
        )

    def test_the_history_holds_the_last_120_values_up_to_the_frame_of_the_value(
        self, field: StarField
    ) -> None:
        state = state_of(300, field)
        assert state.focus is not None
        history = state.focus.history
        assert history is not None
        assert FOCUS_POINTS == 120
        assert len(history.index) == 120
        assert history.index[-1] == state.focus.frame_seq
        assert history.index == list(range(history.index[0], history.index[0] + 120))
        assert history.fwhm_px[-1] == state.focus.fwhm_px
        assert history.t_utc_ms == sorted(history.t_utc_ms)
        lengths = {len(getattr(history, name)) for name in ("seq", "fwhm_px", "n_stars", "spike")}
        assert lengths == {120}

    def test_an_early_frame_has_a_short_history(self, field: StarField) -> None:
        history = state_of(10, field).focus.history  # type: ignore[union-attr]
        assert history is not None
        assert history.index == list(range(1, len(history.index) + 1))
        assert len(history.index) < 120

    def test_every_97th_frame_is_a_spike_and_the_rule_is_the_one_of_core(self) -> None:
        assert FOCUS_SPIKE_EVERY == 97
        last = 400
        flags = demo_spike_flags(1, last)
        reference = FocusHistory()
        expected = []
        for seq in range(1, last + 1):
            point = reference.add(seq, seq, demo_fwhm_px(seq), 30)
            assert point is not None
            expected.append(point.spike)
        assert flags == expected
        assert [seq for seq, spike in enumerate(flags, start=1) if spike] == [97, 194, 291, 388]

    def test_the_value_of_a_spike_frame_carries_the_flag_and_not_the_best_value(
        self, field: StarField
    ) -> None:
        state = next(
            found
            for found in (state_of(seq, field) for seq in range(97, 104))
            if found.focus is not None and found.focus.frame_seq == 97
        )
        assert state.focus is not None
        assert state.focus.spike is True
        assert state.focus.fwhm_px is not None
        assert state.focus.best_fwhm_px is not None
        assert state.focus.fwhm_px > 2.0 * state.focus.best_fwhm_px

    def test_the_best_value_is_the_smallest_value_that_is_no_spike(self, field: StarField) -> None:
        state = state_of(500, field)
        assert state.focus is not None
        assert state.focus.frame_seq is not None
        reference = FocusHistory()
        for seq in range(1, state.focus.frame_seq + 1):
            reference.add(seq, seq, demo_fwhm_px(seq), 30)
        assert state.focus.best_fwhm_px == reference.best_px

    def test_the_reset_restarts_the_best_value_and_keeps_the_history(self) -> None:
        field = StarField(stars=20)
        before = field.frame(200).state.focus
        assert before is not None
        field.reset_focus(200)
        after = field.frame(210).state.focus
        assert after is not None
        assert after.frame_seq is not None
        values = [demo_fwhm_px(seq) for seq in range(200, after.frame_seq + 1)]
        flags = demo_spike_flags(200, after.frame_seq)
        counted = [value for value, spike in zip(values, flags, strict=True) if not spike]
        assert after.best_fwhm_px == min(counted)
        assert after.history is not None
        assert len(after.history.index) == 120  # the curve before the reset stays

    def test_the_demo_core_restarts_the_best_value_at_the_next_frame(self) -> None:
        core = DemoCore(field=StarField(stars=20))
        core._seq = 40
        core.alignment_reset_focus()
        assert core.focus_resets == 1
        assert core._field.focus_reset_seq == 41

    def test_the_focus_stays_small_with_its_history(self, field: StarField) -> None:
        state = state_of(300, field)
        assert state.focus is not None
        assert len(state.focus.model_dump_json()) < 7000


def test_the_states_survive_the_json_and_fit_in_the_message(field: StarField) -> None:
    for seq in (20, FIRST_LOST + 5):
        frame = field.frame(seq)
        state = frame.state
        assert decode_alignment_state(json.loads(state.model_dump_json())) == state
        assert len(pack_frame(state, frame.jpeg)) < MAX_STATE_BYTES + len(frame.jpeg)
        assert len(state.model_dump_json()) < MAX_STATE_BYTES // 8
