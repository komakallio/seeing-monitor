"""The fault tracker: backoff, the recovery ladder, degraded, and recovery."""

from __future__ import annotations

import pytest

from seeingmon.drivers.base import (
    CameraConfigError,
    CameraDisconnectedError,
    CameraError,
    CameraLinkError,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)
from seeingmon.scheduler.config import FaultConfig, LadderConfig
from seeingmon.scheduler.faults import FaultCause, FaultTracker, classify, reason_text
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


def cause_of(faults: FaultTracker) -> FaultCause | None:
    return faults.cause


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

    def test_the_quick_steps_keep_the_backoff_whether_or_not_the_status_is_degraded(self) -> None:
        faults = tracker(degraded_after=3, slow_retry_s=900.0, backoff_initial_s=1.0)
        plans = [faults.failure(0, can_escalate=True) for _ in range(8)]
        assert [plan.step for plan in plans] == [RC, RC, RO, RO, UR, UR, RA, RA]
        assert [plan.wait_s for plan in plans] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        assert [plan.degraded for plan in plans] == [
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ]

    def test_the_wait_is_the_slow_retry_period_once_the_quick_steps_have_had_their_attempts(
        self,
    ) -> None:
        faults = tracker(degraded_after=3, slow_retry_s=900.0, backoff_initial_s=1.0)
        plans = [faults.failure(0, can_escalate=True) for _ in range(12)]
        assert [plan.wait_s for plan in plans[8:]] == [900.0, 900.0, 900.0, 900.0]
        assert [plan.step for plan in plans[8:]] == [RB, RB, PC, PC]

    def test_a_ladder_that_stops_early_slows_down_after_its_last_step(self) -> None:
        capped = tracker(max_level="reopen", degraded_after=100, slow_retry_s=900.0)
        waits = [capped.failure(0, can_escalate=True).wait_s for _ in range(6)]
        assert waits == [2.0, 4.0, 8.0, 16.0, 900.0, 900.0]
        driver_only = tracker(degraded_after=100, slow_retry_s=900.0)
        waits = [driver_only.failure(0, can_escalate=False).wait_s for _ in range(8)]
        assert waits == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 900.0, 900.0]

    def test_a_reboot_that_is_too_soon_still_waits_the_slow_period(self) -> None:
        faults = tracker(attempts=1, degraded_after=100, interval_s=3600.0, slow_retry_s=900.0)
        for _ in range(4):
            faults.failure(0, can_escalate=True)
        first = faults.failure(0, can_escalate=True)
        assert (first.step, first.wait_s) == (RB, 900.0)
        faults.note_destructive(0)
        second = faults.failure(1800 * NS, can_escalate=True)
        assert (second.step, second.wait_s) == (RA, 900.0)

    def test_degraded_changed_marks_only_the_failure_that_degrades(self) -> None:
        faults = tracker(degraded_after=3)
        changed = [faults.failure(0, can_escalate=True).degraded_changed for _ in range(5)]
        assert changed == [False, False, True, False, False]

    def test_a_huge_failure_count_does_not_overflow_the_backoff(self) -> None:
        faults = tracker(attempts=10**6, degraded_after=10**9, backoff_max_s=60.0)
        for _ in range(3000):
            plan = faults.failure(0, can_escalate=True)
        assert plan.wait_s == 60.0
        assert plan.failures == 3000
        assert plan.step is RC  # one million attempts for the first step


class TestTheCause:
    @pytest.mark.parametrize(
        ("error", "cause"),
        [
            (CameraTimeoutError("no frame"), FaultCause.TIMEOUT),
            (CameraDisconnectedError("no ASI camera is connected"), FaultCause.DISCONNECTED),
            (CameraLinkError("cannot reach acquire"), FaultCause.LINK),
            (CameraStateError("the camera is not open"), FaultCause.ERROR),
            (CameraConfigError("a geometry that differs"), FaultCause.ERROR),
            (CameraError("ASISetControlValue failed"), FaultCause.ERROR),
            (RuntimeError("the supervisor failed"), FaultCause.ERROR),
        ],
    )
    def test_each_error_has_a_cause(self, error: Exception, cause: FaultCause) -> None:
        assert classify(error) is cause

    def test_a_link_error_is_also_a_disconnected_error_for_the_code_that_does_not_ask(self) -> None:
        assert issubclass(CameraLinkError, CameraDisconnectedError)
        assert classify(CameraLinkError("x")) is FaultCause.LINK  # the nearer class decides

    def test_a_camera_that_is_not_connected_degrades_the_status_at_once(self) -> None:
        faults = tracker(degraded_after=5)
        plan = faults.failure(0, can_escalate=True, cause=FaultCause.DISCONNECTED)
        assert plan.failures == 1
        assert (plan.degraded, plan.degraded_changed) == (True, True)
        assert faults.degraded
        again = faults.failure(0, can_escalate=True, cause=FaultCause.DISCONNECTED)
        assert (again.degraded, again.degraded_changed) == (True, False)

    @pytest.mark.parametrize("cause", [FaultCause.TIMEOUT, FaultCause.LINK, FaultCause.ERROR])
    def test_any_other_cause_waits_for_the_failure_count(self, cause: FaultCause) -> None:
        faults = tracker(degraded_after=3)
        degraded = [faults.failure(0, can_escalate=True, cause=cause).degraded for _ in range(3)]
        assert degraded == [False, False, True]

    def test_the_plan_names_the_cause_of_the_episode(self) -> None:
        faults = tracker(degraded_after=100)
        causes = [
            faults.failure(0, can_escalate=True, cause=cause).cause
            for cause in (
                FaultCause.TIMEOUT,
                FaultCause.ERROR,  # a failing step does not replace the better explanation
                FaultCause.ERROR,
                FaultCause.LINK,  # acquire is unreachable, which says more
                FaultCause.TIMEOUT,
                FaultCause.DISCONNECTED,  # the camera is not there, which says the most
                FaultCause.LINK,
                FaultCause.ERROR,
            )
        ]
        assert causes == [
            FaultCause.TIMEOUT,
            FaultCause.TIMEOUT,
            FaultCause.TIMEOUT,
            FaultCause.LINK,
            FaultCause.LINK,
            FaultCause.DISCONNECTED,
            FaultCause.DISCONNECTED,
            FaultCause.DISCONNECTED,
        ]

    def test_a_step_that_works_ends_the_explanation_of_the_failures_before_it(self) -> None:
        faults = tracker(degraded_after=100)
        faults.failure(0, can_escalate=True, cause=FaultCause.DISCONNECTED)
        faults.step_succeeded()
        plan = faults.failure(0, can_escalate=True, cause=FaultCause.TIMEOUT)
        assert plan.cause is FaultCause.TIMEOUT

    def test_the_cause_is_unknown_before_the_first_failure_and_after_the_recovery(self) -> None:
        faults = tracker(degraded_after=2, clear_after_frames=1)
        assert cause_of(faults) is None
        faults.failure(0, can_escalate=True, cause=FaultCause.TIMEOUT)
        assert cause_of(faults) is FaultCause.TIMEOUT
        assert faults.success() is True
        assert cause_of(faults) is None


class TestTheReasonInWords:
    def test_a_timeout_says_that_the_camera_may_be_disconnected(self) -> None:
        text = reason_text(FaultCause.TIMEOUT, CameraTimeoutError("no frame arrived in time"))
        assert text == "no frame arrived; the camera may be disconnected"

    def test_a_lost_camera_says_so(self) -> None:
        error = CameraDisconnectedError("no ASI camera is connected")
        assert reason_text(FaultCause.DISCONNECTED, error) == "the camera is not connected"

    def test_a_broken_link_names_the_process(self) -> None:
        error = CameraLinkError("cannot reach acquire")
        assert reason_text(FaultCause.LINK, error) == "the camera process (acquire) does not answer"

    def test_another_error_carries_its_own_message(self) -> None:
        error = CameraError("ASISetControlValue failed with GENERAL_ERROR (16).")
        assert reason_text(FaultCause.ERROR, error) == (
            "the camera reported an error: ASISetControlValue failed with GENERAL_ERROR (16)"
        )

    def test_a_long_message_is_cut(self) -> None:
        text = reason_text(FaultCause.ERROR, CameraError("x" * 1000))
        assert len(text) < 300
        assert text.endswith("...")


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
