"""The state machine of the scheduler: the states, the legal transitions, and the history.

- `safe`: the camera is idle, with a brightness watch. The scheduler enters it at start, in
  daylight, when the sky is too bright, after a persistent fault, and after `align`, `paused`,
  or a commissioning task that interrupted `safe`.
- `auto`: the scheduler repeats a fast window and a survey step. It enters this state when the
  sky is dark enough and the camera works.
- `align`: the alignment stream preempts everything. `StartAlignment` enters it, and it ends on
  `StopAlignment` or after the idle timeout.
- `commission`: the registered handlers run the queued tasks. The scheduler enters it when a
  task waits and the cycle reaches a boundary.
- `paused`: nothing runs. `Pause` enters it.

The machine holds only the logical state. It is not thread-safe, and the scheduler guards it
with a lock. Which state follows which is the whole policy, so the table `LEGAL_TRANSITIONS`
is the one place to read it. `commission` returns to the state that it interrupted, which is
`safe` or `auto`. `align` and `paused` always go back through `safe`, so the sky check runs again
before the scheduler resumes `auto`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class State(StrEnum):
    """The five states. The values are the codes of `HEALTH_STATES` in the health record."""

    SAFE = "safe"
    AUTO = "auto"
    ALIGN = "align"
    COMMISSION = "commission"
    PAUSED = "paused"


LEGAL_TRANSITIONS: Mapping[State, frozenset[State]] = {
    State.SAFE: frozenset({State.AUTO, State.ALIGN, State.COMMISSION, State.PAUSED}),
    State.AUTO: frozenset({State.SAFE, State.ALIGN, State.COMMISSION, State.PAUSED}),
    State.ALIGN: frozenset({State.SAFE, State.PAUSED}),
    State.COMMISSION: frozenset({State.SAFE, State.AUTO, State.ALIGN, State.PAUSED}),
    State.PAUSED: frozenset({State.SAFE}),
}


class IllegalTransitionError(RuntimeError):
    """The code asked for a transition that `LEGAL_TRANSITIONS` does not list."""


@dataclass(frozen=True, slots=True)
class Transition:
    """One change of state. `reason` is a short code or phrase for the event record."""

    t_utc_ns: int
    from_state: State
    to_state: State
    reason: str


class StateMachine:
    """The current state, when it began, and a bounded history of transitions."""

    def __init__(self, initial: State, *, now_utc_ns: int, reason: str = "start") -> None:
        self._state = initial
        self._reason = reason
        self._entered_utc_ns = now_utc_ns
        self._last_transition_utc_ns: int | None = None
        self._count = 0
        self._history: deque[Transition] = deque(maxlen=100)

    @property
    def state(self) -> State:
        return self._state

    @property
    def reason(self) -> str:
        """Why the machine entered the current state."""
        return self._reason

    @property
    def entered_utc_ns(self) -> int:
        """When the machine entered the current state."""
        return self._entered_utc_ns

    @property
    def last_transition_utc_ns(self) -> int | None:
        """The time of the latest transition, or `None` before the first."""
        return self._last_transition_utc_ns

    @property
    def transition_count(self) -> int:
        return self._count

    @property
    def history(self) -> tuple[Transition, ...]:
        """The latest transitions (up to 100), oldest first."""
        return tuple(self._history)

    def can_transition(self, to_state: State) -> bool:
        return to_state in LEGAL_TRANSITIONS[self._state]

    def transition(self, to_state: State, reason: str, now_utc_ns: int) -> Transition:
        """Move to `to_state`. Raises `IllegalTransitionError` when the table forbids it."""
        if not self.can_transition(to_state):
            raise IllegalTransitionError(f"cannot go from {self._state} to {to_state}")
        change = Transition(now_utc_ns, self._state, to_state, reason)
        self._state = to_state
        self._reason = reason
        self._entered_utc_ns = now_utc_ns
        self._last_transition_utc_ns = now_utc_ns
        self._count += 1
        self._history.append(change)
        return change
