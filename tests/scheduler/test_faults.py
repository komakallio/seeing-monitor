"""The fault tracker: backoff, the recovery ladder, degraded, and recovery."""

from __future__ import annotations

import pytest

from seeingmon.drivers.base import RecoveryLevel
from seeingmon.scheduler.config import FaultConfig, LadderConfig
from seeingmon.scheduler.faults import FaultTracker
from seeingmon.scheduler.levels import (
    LADDER,
    STEP_NAMES,
    EscalationLevel,
    parse_step,
    step_name,
)

NS = 1_000_000_000
RC, RO, UR = RecoveryLevel.RESTART_CAPTURE, RecoveryLevel.REOPEN, RecoveryLevel.USB_RESET
RA, RB, PC = (
    EscalationLevel.RESTART_ACQUIRE,
    EscalationLevel.REBOOT,
    EscalationLevel.POWER_CYCLE,
)


def tracker(
    *,
    attempts: int = 2,
    max_level: str = "power_cycle",
    interval_s: float = 21_600.0,
    **faults: float,
) -> FaultTracker:
    return FaultTracker(
        FaultConfig(**faults),
        LadderConfig(
            attempts_per_level=attempts, max_level=max_level, destructive_interval_s=interval_s
        ),
    )


class TestLadderLevels:
    def test_the_ladder_runs_from_the_driver_steps_to_the_supervisor_steps(self) -> None:
        assert list(LADDER) == [RC, RO, UR, RA, RB, PC]
        assert [int(step) for step in LADDER] == [1, 2, 3, 4, 5, 6]

    def test_the_names_round_trip(self) -> None:
        assert STEP_NAMES == (
            "restart_capture",
            "reopen",
            "usb_reset",
            "restart_acquire",
            "reboot",
            "power_cycle",
        )
        for step in LADDER:
            assert parse_step(step_name(step)) is step

    def test_an_unknown_name_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unknown recovery step"):
            parse_step("hammer")


class TestFailurePlan:
    def test_the_steps_climb_in_order_and_repeat_the_top_step(self) -> None:
        faults = tracker(attempts=2, degraded_after=100)
        steps = [faults.failure(0, can_escalate=True).step for _ in range(14)]
        assert steps == [RC, RC, RO, RO, UR, UR, RA, RA, RB, RB, PC, PC, PC, PC]

    def test_a_single_attempt_per_level_climbs_one_step_per_failure(self) -> None:
        faults = tracker(attempts=1, degraded_after=100, interval_s=0.0)
        steps = [faults.failure(0, can_escalate=True).step for _ in range(8)]
        assert steps == [RC, RO, UR, RA, RB, PC, PC, PC]

    def test_max_level_limits_the_climb(self) -> None:
        faults = tracker(attempts=1, max_level="usb_reset", degraded_after=100)
        assert [faults.failure(0, can_escalate=True).step for _ in range(6)] == [
            RC,
            RO,
            UR,
            UR,
            UR,
            UR,
        ]

    def test_without_a_supervisor_the_ladder_stops_at_the_last_driver_step(self) -> None:
        faults = tracker(attempts=1, degraded_after=100)
        assert [faults.failure(0, can_escalate=False).step for _ in range(6)] == [
            RC,
            RO,
            UR,
            UR,
            UR,
            UR,
        ]

    def test_a_reboot_or_power_cycle_waits_for_the_destructive_interval(self) -> None:
        faults = tracker(attempts=1, degraded_after=100, interval_s=3600.0)
        for _ in range(4):
            faults.failure(0, can_escalate=True)
        first = faults.failure(0, can_escalate=True)
        assert first.step is RB  # the first reboot is free
        faults.note_destructive(0)
        # Within the hour the scheduler falls back to the highest gentler step.
        assert faults.failure(1800 * NS, can_escalate=True).step is RA
        assert faults.failure(3599 * NS, can_escalate=True).step is RA
        # After the hour the top step is allowed again.
        assert faults.failure(3600 * NS, can_escalate=True).step is PC

    def test_the_wait_doubles_up_to_the_maximum(self) -> None:
        faults = tracker(
            backoff_initial_s=2.0, backoff_factor=2.0, backoff_max_s=20.0, degraded_after=100
        )
        waits = [faults.failure(0, can_escalate=True).wait_s for _ in range(7)]
        assert waits == [2.0, 4.0, 8.0, 16.0, 20.0, 20.0, 20.0]

    def test_a_degraded_status_waits_for_the_slow_retry_period(self) -> None:
        faults = tracker(degraded_after=3, slow_retry_s=900.0, backoff_initial_s=1.0)
        plans = [faults.failure(0, can_escalate=True) for _ in range(5)]
        assert [plan.wait_s for plan in plans] == [1.0, 2.0, 900.0, 900.0, 900.0]
        assert [plan.degraded for plan in plans] == [False, False, True, True, True]

    def test_degraded_changed_marks_only_the_failure_that_degrades(self) -> None:
        faults = tracker(degraded_after=3)
        changed = [faults.failure(0, can_escalate=True).degraded_changed for _ in range(5)]
        assert changed == [False, False, True, False, False]

    def test_a_huge_failure_count_does_not_overflow_the_backoff(self) -> None:
        faults = tracker(degraded_after=10**9, backoff_max_s=60.0)
        for _ in range(3000):
            plan = faults.failure(0, can_escalate=True)
        assert plan.wait_s == 60.0
        assert plan.failures == 3000


class TestRecovery:
    def test_good_frames_without_a_failure_change_nothing(self) -> None:
        faults = tracker()
        assert faults.success() is False
        assert faults.good_frames == 0

    def test_enough_good_frames_in_a_row_clear_the_failures_and_degraded(self) -> None:
        faults = tracker(degraded_after=2, clear_after_frames=3)
        faults.failure(0, can_escalate=True)
        faults.failure(0, can_escalate=True)
        assert faults.degraded
        assert faults.success() is False
        assert faults.success() is False
        assert faults.degraded
        assert faults.success() is True
        assert (faults.failures, faults.degraded, faults.good_frames) == (0, False, 0)

    def test_a_failure_restarts_the_count_of_good_frames(self) -> None:
        faults = tracker(degraded_after=2, clear_after_frames=3)
        faults.failure(0, can_escalate=True)
        faults.success()
        faults.success()
        faults.failure(0, can_escalate=True)
        assert faults.good_frames == 0
        assert faults.failures == 2
        faults.success()
        faults.success()
        assert faults.degraded  # two good frames are not enough after the second failure
        assert faults.success() is True

    def test_after_recovery_the_ladder_starts_from_the_bottom_again(self) -> None:
        faults = tracker(attempts=1, degraded_after=100, clear_after_frames=1)
        for _ in range(3):
            faults.failure(0, can_escalate=True)
        faults.success()
        assert faults.failure(0, can_escalate=True).step is RC
