"""Camera faults: backoff, the recovery ladder, the escalation callback, degraded, and recovery.

A fault makes every read time out. The ladder runs in this order: restart the capture, reopen
the camera, reset the USB device (all three through `CameraDriver.recover`), then restart `acquire`,
reboot, and power cycle (through the `escalate` callback). These scenarios use one attempt for each
step and short waits, so that a ladder of six steps fits in a few minutes of virtual time.
"""

from __future__ import annotations

import itertools

import pytest

from seeingmon.clock import iso_to_utc_ns
from seeingmon.drivers.base import CameraDisconnectedError, CameraError, RecoveryLevel
from seeingmon.scheduler.config import FaultConfig, LadderConfig, LoopConfig, SchedulerConfig
from seeingmon.scheduler.levels import EscalationLevel
from tests.scheduler.scenario import TEST_CONFIG, World

NIGHT = iso_to_utc_ns("2026-01-01T22:00:00Z")
RC, RO, UR = RecoveryLevel.RESTART_CAPTURE, RecoveryLevel.REOPEN, RecoveryLevel.USB_RESET
RA, RB, PC = EscalationLevel.RESTART_ACQUIRE, EscalationLevel.REBOOT, EscalationLevel.POWER_CYCLE


def quick_config(**ladder: float | str) -> SchedulerConfig:
    """One attempt for each step, short waits, and degraded after three failures."""
    return SchedulerConfig(
        fast=TEST_CONFIG.fast,
        loop=LoopConfig(max_sleep_s=5.0),
        faults=FaultConfig(
            backoff_initial_s=1.0,
            backoff_factor=2.0,
            backoff_max_s=8.0,
            degraded_after=3,
            slow_retry_s=60.0,
            clear_after_frames=5,
        ),
        ladder=LadderConfig(attempts_per_level=1, destructive_interval_s=0.0, **ladder),
    )


def recover_levels(world: World) -> list[object]:
    return [level for _, level in world.camera.calls_named("recover")]


@pytest.fixture(scope="module")
def outage_world() -> World:
    """The camera is out from 1000 to 1010, and it works again by itself.

    A read that starts at 1000 times out after 4.5 seconds. The scheduler waits 1 second, restarts
    the capture, and the next read starts at 1005.5, which is still inside the outage. It waits
    2 seconds, reopens the camera, and the read that starts at 1012 works.
    """
    world = World(start_utc_ns=NIGHT, config=quick_config())
    world.camera_fault(1000, 1010)
    world.run_until(2000)
    world.close()
    return world


@pytest.fixture(scope="module")
def persistent_world() -> World:
    """Only a restart of `acquire` clears this fault. The first three steps do not."""
    world = World(start_utc_ns=NIGHT, config=quick_config())
    world.camera_fault(1000, None, fixed_by=int(RA))
    world.run_until(4000)
    world.close()
    return world


class TestAShortOutage:
    def test_each_failed_read_is_a_fault_event_followed_by_a_step_of_the_ladder(
        self, outage_world: World
    ) -> None:
        faults = outage_world.events("scheduler.fault")
        steps = outage_world.events("scheduler.recovery_step")
        assert len(faults) == 2
        assert [(e.detail or {})["step"] for e in steps] == ["restart_capture", "reopen"]
        assert all((e.detail or {})["ok"] is True for e in steps)
        assert [(e.detail or {})["failures"] for e in faults] == [1, 2]

    def test_the_driver_steps_run_in_the_order_of_the_ladder(self, outage_world: World) -> None:
        assert recover_levels(outage_world) == [RC, RO]
        assert outage_world.escalations == []

    def test_the_waits_double(self, outage_world: World) -> None:
        waits = [(e.detail or {})["wait_s"] for e in outage_world.events("scheduler.fault")]
        assert waits == [1.0, 2.0]

    def test_the_fault_events_say_when_and_where(self, outage_world: World) -> None:
        first, second = outage_world.events("scheduler.fault")
        assert outage_world.seconds(first.t_utc_ns) == pytest.approx(1004.5, abs=0.1)
        assert outage_world.seconds(second.t_utc_ns) == pytest.approx(1010.0, abs=0.1)
        assert (first.detail or {})["where"] == "reading a fast frame"
        assert "scripted fault" in (first.detail or {})["error"]

    def test_the_scheduler_recovers_and_the_stream_resumes_in_a_new_stream(
        self, outage_world: World
    ) -> None:
        assert outage_world.states_visited() == ["safe", "auto"]  # it never left auto
        assert outage_world.scheduler.state.value == "auto"
        assert outage_world.scheduler.status().degraded is False
        starts = [
            outage_world.seconds(c.t_utc_ns)
            for c in outage_world.configures(mode="bin1", video=True)
        ]
        assert [round(t) for t in starts if 1000 < t < 1100] == [1006, 1012]  # a stream per try
        assert outage_world.windows()[-1].t_utc_ns > outage_world.t(1100)

    def test_the_failure_count_clears_after_enough_good_frames(self, outage_world: World) -> None:
        (recovered,) = outage_world.events("scheduler.recovered")
        assert outage_world.seconds(recovered.t_utc_ns) > 1012
        status = outage_world.scheduler.status()
        assert status.fault.failures == 0
        assert status.fault.last_error is not None  # the last error stays for the operator
        assert outage_world.events("scheduler.degraded") == []

    def test_no_window_is_lost_across_the_fault(self, outage_world: World) -> None:
        frames_in_windows = sum(w.n_frames for w in outage_world.windows())
        assert frames_in_windows == outage_world.fast.frames_pushed

    def test_every_event_of_the_episode_has_the_right_level(self, outage_world: World) -> None:
        for event in outage_world.events("scheduler.fault"):
            assert event.level == "warning"  # not degraded, so not yet an error
        assert outage_world.events("scheduler.recovered")[0].level == "info"


class TestAPersistentFault:
    def test_the_ladder_climbs_through_the_driver_steps_to_the_supervisor(
        self, persistent_world: World
    ) -> None:
        world = persistent_world
        assert recover_levels(world) == [RC, RO, UR]
        assert [level for _, level in world.escalations] == [RA]
        assert [(e.detail or {})["step"] for e in world.events("scheduler.recovery_step")] == [
            "restart_capture",
            "reopen",
            "usb_reset",
            "restart_acquire",
        ]

    def test_the_status_turns_degraded_after_repeated_failures_and_the_scheduler_goes_safe(
        self, persistent_world: World
    ) -> None:
        world = persistent_world
        (degraded,) = world.events("scheduler.degraded")
        assert degraded.level == "error"
        assert (degraded.detail or {})["failures"] == 3
        assert world.states_visited() == ["safe", "auto", "safe", "auto"]
        reasons = [(e.detail or {})["reason"] for e in world.events("scheduler.state_change")]
        assert reasons == ["the sky is dark enough", "camera fault", "the sky is dark enough"]

    def test_while_degraded_the_scheduler_retries_at_the_slow_pace(
        self, persistent_world: World
    ) -> None:
        faults = persistent_world.events("scheduler.fault")
        assert [(e.detail or {})["failures"] for e in faults] == [1, 2, 3, 4]
        waits = [(e.detail or {})["wait_s"] for e in faults]
        assert waits == [1.0, 2.0, 60.0, 60.0]  # the third failure degrades, so the wait is slow

    def test_a_supervisor_step_sends_the_escalation_and_reopens_the_camera(
        self, persistent_world: World
    ) -> None:
        world = persistent_world
        assert len(world.camera.calls_named("open")) == 2  # at the start, and after `acquire`
        time_of_escalation = world.escalations[0][0]
        assert world.t(1000) < time_of_escalation < world.t(1400)
        (event,) = [
            e for e in world.events("scheduler.recovery_step") if (e.detail or {})["level"] == 4
        ]
        assert event.level == "warning"  # a supervisor step is more serious than a driver step

    def test_after_the_fix_good_frames_clear_degraded_and_auto_resumes(
        self, persistent_world: World
    ) -> None:
        world = persistent_world
        assert world.scheduler.state.value == "auto"
        status = world.scheduler.status()
        assert status.degraded is False
        assert status.fault.failures == 0
        assert len(world.events("scheduler.recovered")) == 1
        assert world.windows()[-1].t_utc_ns > world.t(3000)

    def test_the_counters_add_up(self, persistent_world: World) -> None:
        counters = persistent_world.scheduler.status().counters
        assert counters.faults == 4
        assert counters.recovery_steps == 4
        assert counters.escalations == 1


class TestTheLadderWithoutASupervisor:
    def test_without_an_escalate_callback_the_ladder_stops_at_the_last_driver_step(self) -> None:
        world = World(start_utc_ns=NIGHT, config=quick_config(), escalate=False)
        world.camera_fault(1000, None)
        world.run_until(1000 + 8 * 60)
        levels = recover_levels(world)
        assert levels[:3] == [RC, RO, UR]
        assert set(levels[3:]) == {UR}  # it keeps trying the last step that it owns
        assert world.escalations == []
        assert world.scheduler.status().degraded is True
        world.close()


class TestDestructiveSteps:
    def test_a_reboot_or_power_cycle_comes_at_most_once_per_interval(self) -> None:
        config = SchedulerConfig(
            fast=TEST_CONFIG.fast,
            loop=LoopConfig(max_sleep_s=5.0),
            faults=FaultConfig(
                backoff_initial_s=1.0, backoff_max_s=2.0, degraded_after=3, slow_retry_s=60.0
            ),
            ladder=LadderConfig(attempts_per_level=1, destructive_interval_s=1800.0),
        )
        world = World(start_utc_ns=NIGHT, config=config)
        world.camera_fault(1000, None)  # nothing fixes it
        world.run_until(1000 + 4 * 3600)
        levels = [level for _, level in world.escalations]
        times = [t for t, level in world.escalations if level in (RB, PC)]
        assert RA in levels
        assert RB in levels  # a reboot happens once the gentler steps have failed
        gaps = [(later - earlier) / 1e9 for earlier, later in itertools.pairwise(times)]
        assert all(gap >= 1800.0 for gap in gaps)
        assert len(times) <= 8  # 4 hours at one per 30 minutes
        # Between destructive steps the scheduler falls back to the restart of `acquire`.
        assert levels.count(RA) > len(times)
        world.close()

    def test_max_level_keeps_the_scheduler_from_asking_for_more(self) -> None:
        world = World(start_utc_ns=NIGHT, config=quick_config(max_level="restart_acquire"))
        world.camera_fault(1000, None)
        world.run_until(1000 + 3600)
        assert {level for _, level in world.escalations} == {RA}
        world.close()


class TestFailingSteps:
    def test_a_recovery_step_that_fails_counts_as_a_failure_and_the_ladder_moves_on(self) -> None:
        world = World(start_utc_ns=NIGHT, config=quick_config())
        world.camera.fail_recover(CameraError("the USB reset did not work"))
        world.camera_fault(1000, None, fixed_by=int(RA))
        world.run_until(3000)
        failed = [e for e in world.events("scheduler.recovery_step") if not (e.detail or {})["ok"]]
        assert len(failed) >= 1
        assert failed[0].level == "error"
        assert "did not work" in failed[0].message
        # The failed step counted, so the ladder reached the supervisor step sooner.
        assert RA in [level for _, level in world.escalations]
        world.close()

    def test_an_escalation_callback_that_raises_is_a_failed_step_and_not_a_crash(self) -> None:
        world = World(start_utc_ns=NIGHT, config=quick_config())

        def broken(level: EscalationLevel) -> None:
            raise RuntimeError("the supervisor is not reachable")

        world.scheduler._escalate = broken  # the scenario's own callback is replaced
        world.camera_fault(1000, None)
        world.run_until(2500)
        failed = [
            e
            for e in world.events("scheduler.recovery_step")
            if not (e.detail or {})["ok"] and (e.detail or {})["level"] == 4
        ]
        assert failed
        assert world.scheduler.state.value == "safe"  # degraded, and still alive
        world.close()


class TestASilentSecondOpen:
    def test_a_camera_that_is_absent_at_the_start_is_retried(self) -> None:
        """No camera answers at first. The ladder runs, and the watch starts when one appears."""
        world = World(start_utc_ns=NIGHT, config=quick_config())
        world.camera.fail_open(CameraDisconnectedError("no camera answers"))
        world.at(300, lambda w: setattr(w.camera, "_open_failure", None))
        world.run_until(1200)
        assert world.events("scheduler.fault")
        assert world.scheduler.state.value == "auto"
        assert world.scheduler.status().degraded is False
        world.close()


class TestStatusDuringAFault:
    def test_the_status_shows_the_next_attempt_and_the_step(self) -> None:
        world = World(start_utc_ns=NIGHT, config=quick_config())
        world.camera_fault(1000, None, fixed_by=int(UR))
        world.run_until(1000)
        # Step until the first fault, then look at the plan.
        while world.scheduler.status().fault.failures == 0:
            world.scheduler.step()
        status = world.scheduler.status()
        assert status.fault.failures == 1
        assert status.fault.next_step == "restart_capture"
        assert status.fault.next_attempt_utc_ns is not None
        assert status.fault.next_attempt_utc_ns > status.t_utc_ns
        assert status.fault.last_error is not None
        assert "scripted fault" in status.fault.last_error
        assert status.camera_component == "degraded"
        world.close()

    def test_the_health_fields_follow_the_degraded_state(self) -> None:
        world = World(start_utc_ns=NIGHT, config=quick_config())
        world.camera_fault(1000, None, fixed_by=int(RA))
        world.run_until(1100)  # degraded since 1016, and the fix comes at about 1137
        fields = world.scheduler.status().health_fields()
        assert fields["state"] == "safe"
        assert fields["degraded"] is True
        assert fields["components"] == {"scheduler": "degraded", "camera": "failed"}
        world.close()
