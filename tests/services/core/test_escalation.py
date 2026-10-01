"""The steps of the recovery ladder that `core` performs: restart acquire, reboot, power cycle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.base import CameraTimeoutError
from seeingmon.hardware.power import CommandResult, PowerOutcome, PowerResult
from seeingmon.records import EventRecord
from seeingmon.scheduler.levels import EscalationLevel
from seeingmon.services.core.escalation import Escalator
from seeingmon.services.core.events import EventWriter
from seeingmon.services.core.settings import EscalationSettings
from seeingmon.testing import ListRecordWriter


class FakeRunner:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], float]] = []
        self.result = result or CommandResult(0)

    def run(self, argv: Sequence[str], timeout_s: float) -> CommandResult:
        self.calls.append((tuple(argv), timeout_s))
        return self.result


class FakePower:
    def __init__(self, outcome: PowerOutcome = PowerOutcome.DONE) -> None:
        self.reasons: list[str] = []
        self.outcome = outcome

    def request(self, reason: str) -> PowerResult:
        self.reasons.append(reason)
        return PowerResult(self.outcome)


def build(
    **parts: object,
) -> tuple[Escalator, ListRecordWriter, VirtualClock]:
    sink = ListRecordWriter()
    clock = VirtualClock(1_800_000_000_000_000_000)
    writer = EventWriter(sink.write, station_id="test", profile_id="p", clock=clock)
    settings = cast(EscalationSettings, parts.pop("settings", EscalationSettings()))
    escalator = Escalator(writer=writer, clock=clock, settings=settings, **parts)  # type: ignore[arg-type]
    return escalator, sink, clock


def kinds(sink: ListRecordWriter) -> list[str]:
    return [r.kind for r in sink.of_type("event") if isinstance(r, EventRecord)]


def detail_of(sink: ListRecordWriter, kind: str) -> dict[str, object]:
    for record in sink.of_type("event"):
        if isinstance(record, EventRecord) and record.kind == kind:
            return record.detail or {}
    raise AssertionError(f"no {kind} event")


class TestRestartAcquire:
    def test_it_asks_and_waits_until_the_connection_is_gone(self) -> None:
        reasons: list[str] = []
        connected = iter([True, True, False])
        escalator, sink, clock = build(
            restart_acquire=reasons.append, acquire_connected=lambda: next(connected)
        )
        before = clock.monotonic_ns()
        escalator(EscalationLevel.RESTART_ACQUIRE)
        assert reasons == ["recovery ladder: restart_acquire"]
        assert kinds(sink) == ["escalation.restart_acquire", "escalation.acquire_stopped"]
        assert clock.monotonic_ns() - before == 400_000_000  # two polls of 0.2 s

    def test_an_acquire_that_stays_connected_is_reported_after_the_wait(self) -> None:
        escalator, sink, _ = build(
            restart_acquire=lambda reason: None,
            acquire_connected=lambda: True,
            settings=EscalationSettings(restart_acquire_wait_s=1.0),
        )
        escalator(EscalationLevel.RESTART_ACQUIRE)
        assert kinds(sink)[-1] == "escalation.acquire_still_connected"
        assert detail_of(sink, "escalation.acquire_still_connected") == {"waited_s": 1.0}

    def test_a_refused_request_is_an_error_event(self) -> None:
        def refuse(reason: str) -> None:
            raise CameraTimeoutError("acquire did not answer restart in time")

        escalator, sink, _ = build(restart_acquire=refuse)
        escalator(EscalationLevel.RESTART_ACQUIRE)
        assert kinds(sink) == ["escalation.restart_acquire", "escalation.restart_acquire_failed"]
        assert detail_of(sink, "escalation.restart_acquire_failed") == {
            "error": "CameraTimeoutError"
        }

    def test_without_a_way_to_ask_the_step_says_so(self) -> None:
        escalator, sink, _ = build()
        escalator(EscalationLevel.RESTART_ACQUIRE)
        assert kinds(sink) == ["escalation.restart_acquire_unavailable"]


class TestReboot:
    def test_no_command_means_no_reboot(self) -> None:
        runner = FakeRunner()
        escalator, sink, _ = build(runner=runner)
        escalator(EscalationLevel.REBOOT)
        assert runner.calls == []
        assert kinds(sink) == ["escalation.reboot_unavailable"]

    def test_the_command_runs_as_a_list_after_the_event_is_written(self) -> None:
        runner = FakeRunner()
        settings = EscalationSettings(reboot_command=["reboot-tool", "--now"], command_timeout_s=7)
        escalator, sink, _ = build(runner=runner, settings=settings)
        escalator(EscalationLevel.REBOOT)
        assert runner.calls == [(("reboot-tool", "--now"), 7.0)]
        assert kinds(sink) == ["escalation.reboot"]
        assert "reboot-tool" not in str([r.model_dump() for r in sink.records])  # no command

    @pytest.mark.parametrize(
        "result", [CommandResult(1), CommandResult(None, timed_out=True)], ids=["exit", "timeout"]
    )
    def test_a_failing_command_is_an_error_event(self, result: CommandResult) -> None:
        settings = EscalationSettings(reboot_command=["reboot-tool"])
        escalator, sink, _ = build(runner=FakeRunner(result), settings=settings)
        escalator(EscalationLevel.REBOOT)
        assert kinds(sink) == ["escalation.reboot", "escalation.reboot_failed"]


class TestPowerCycle:
    def test_the_hook_gets_a_reason_without_secrets(self) -> None:
        power = FakePower()
        escalator, sink, _ = build(power=power)
        escalator(EscalationLevel.POWER_CYCLE)
        assert power.reasons == ["recovery ladder: power_cycle"]
        assert kinds(sink) == ["escalation.power_cycle", "escalation.power_cycle_result"]
        assert detail_of(sink, "escalation.power_cycle_result") == {"outcome": "done"}

    def test_a_blocked_cycle_is_a_warning(self) -> None:
        escalator, sink, _ = build(power=FakePower(PowerOutcome.RATE_LIMITED))
        escalator(EscalationLevel.POWER_CYCLE)
        result = [
            r
            for r in sink.of_type("event")
            if getattr(r, "kind", "") == "escalation.power_cycle_result"
        ]
        assert [getattr(r, "level", "") for r in result] == ["warning"]

    def test_without_a_hook_the_step_says_so(self) -> None:
        escalator, sink, _ = build()
        escalator(EscalationLevel.POWER_CYCLE)
        assert kinds(sink) == ["escalation.power_cycle_unavailable"]


class TestRobustness:
    def test_a_step_that_raises_is_reported_and_never_propagates(self) -> None:
        def explode(reason: str) -> None:
            raise RuntimeError("boom")

        escalator, sink, _ = build(restart_acquire=explode)
        escalator(EscalationLevel.RESTART_ACQUIRE)
        assert kinds(sink)[-1] == "escalation.failed"

    def test_the_performed_steps_are_listed_in_order(self) -> None:
        escalator, _, _ = build(restart_acquire=lambda reason: None)
        escalator(EscalationLevel.RESTART_ACQUIRE)
        escalator(EscalationLevel.REBOOT)
        escalator(EscalationLevel.POWER_CYCLE)
        assert escalator.performed == [
            EscalationLevel.RESTART_ACQUIRE,
            EscalationLevel.REBOOT,
            EscalationLevel.POWER_CYCLE,
        ]
