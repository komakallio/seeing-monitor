"""The state machine: the legal transitions and the history."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.records.system import HEALTH_STATES
from seeingmon.scheduler.machine import (
    LEGAL_TRANSITIONS,
    IllegalTransitionError,
    State,
    StateMachine,
)

S, A, L, C, P = State.SAFE, State.AUTO, State.ALIGN, State.COMMISSION, State.PAUSED


def test_the_states_are_the_documented_health_codes() -> None:
    assert {state.value for state in State} == set(HEALTH_STATES)


def test_the_machine_starts_in_the_given_state_with_no_history() -> None:
    machine = StateMachine(S, now_utc_ns=1000, reason="start")
    assert machine.state is S
    assert machine.reason == "start"
    assert machine.entered_utc_ns == 1000
    assert machine.last_transition_utc_ns is None
    assert machine.transition_count == 0
    assert machine.history == ()


# The policy, written out. A pause or an alignment never goes straight to `auto`, so the sky
# check runs first. A commissioning task returns to what it interrupted.
EXPECTED = {
    (S, A),
    (S, L),
    (S, C),
    (S, P),
    (A, S),
    (A, L),
    (A, C),
    (A, P),
    (L, S),
    (L, P),
    (C, S),
    (C, A),
    (C, L),
    (C, P),
    (P, S),
}


@pytest.mark.parametrize(("source", "target"), list(itertools.product(State, State)))
def test_the_transition_table_matches_the_policy(source: State, target: State) -> None:
    machine = StateMachine(source, now_utc_ns=0)
    legal = (source, target) in EXPECTED
    assert machine.can_transition(target) is legal
    assert (target in LEGAL_TRANSITIONS[source]) is legal
    if legal:
        assert machine.transition(target, "test", 5).to_state is target
    else:
        with pytest.raises(IllegalTransitionError):
            machine.transition(target, "test", 5)
        assert machine.state is source  # a refused transition changes nothing
        assert machine.transition_count == 0


def test_a_transition_records_the_time_the_reason_and_the_history() -> None:
    machine = StateMachine(S, now_utc_ns=0)
    change = machine.transition(A, "dark", 100)
    machine.transition(C, "task", 250)
    assert (change.from_state, change.to_state, change.reason, change.t_utc_ns) == (
        S,
        A,
        "dark",
        100,
    )
    assert machine.state is C
    assert machine.reason == "task"
    assert machine.entered_utc_ns == 250
    assert machine.last_transition_utc_ns == 250
    assert machine.transition_count == 2
    assert [(t.from_state, t.to_state) for t in machine.history] == [(S, A), (A, C)]


def test_the_history_keeps_the_latest_hundred_transitions() -> None:
    machine = StateMachine(S, now_utc_ns=0)
    for step in range(250):
        machine.transition(A if machine.state is S else S, "flip", step)
    assert machine.transition_count == 250
    assert len(machine.history) == 100
    assert machine.history[-1].t_utc_ns == 249


def test_every_state_can_be_left_and_entered() -> None:
    for state in State:
        assert LEGAL_TRANSITIONS[state], f"{state} is a dead end"
        assert any(state in targets for targets in LEGAL_TRANSITIONS.values()), (
            f"{state} is unreachable"
        )


def test_a_state_never_transitions_to_itself() -> None:
    assert all(state not in targets for state, targets in LEGAL_TRANSITIONS.items())


@given(st.lists(st.sampled_from(list(State)), max_size=60))
def test_a_random_walk_never_leaves_a_legal_state(requests: list[State]) -> None:
    machine = StateMachine(S, now_utc_ns=0)
    for step, target in enumerate(requests):
        before = machine.state
        if machine.can_transition(target):
            machine.transition(target, "walk", step)
            assert machine.state is target
        else:
            with pytest.raises(IllegalTransitionError):
                machine.transition(target, "walk", step)
            assert machine.state is before
        assert machine.state in State
