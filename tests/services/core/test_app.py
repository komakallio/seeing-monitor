"""`CoreApp`: the wiring, the records, the periodic work, the escalation, and the shutdown."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import ClockStatus, VirtualClock
from seeingmon.config import ConfigError
from seeingmon.drivers.base import CameraTimeoutError
from seeingmon.hardware.events import HardwareEvent
from seeingmon.records import HealthRecord, RunRecord
from seeingmon.scheduler.levels import EscalationLevel
from seeingmon.services.config import ServicesConfig
from seeingmon.services.core.app import EXIT_THREAD_DIED, CoreApp, CoreParts
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.notify import SystemdNotifier
from seeingmon.store.db import Store
from seeingmon.testing import FakeCameraDriver

from .rig import NIGHT, CoreRig, build_rig, events_of, make_config, read_all

SLOW_RETRY = "[scheduler.faults]" + chr(10) + "slow_retry_s = 5.0" + chr(10)


@pytest.fixture
def rig(tmp_path: Path) -> CoreRig:
    built = build_rig(tmp_path)
    built.app.start()
    return built


@pytest.fixture(autouse=True)
def _stop_apps(request: pytest.FixtureRequest) -> Any:
    yield
    rig = request.node.funcargs.get("rig")
    if rig is not None:
        rig.app.stop()


class TestStartup:
    def test_the_run_record_waits_for_the_camera_and_names_it(self, rig: CoreRig) -> None:
        assert rig.records("run") == []  # nothing yet: the scheduler has not opened the camera
        rig.app.scheduler.step()  # the first step opens the camera
        (record,) = rig.records("run")
        assert isinstance(record, RunRecord)
        assert record.versions["camera_model"] == "Fake camera"
        assert record.station_id == "test"
        assert record.t_utc_ns == rig.clock.utc_ns() or record.t_utc_ns >= NIGHT
        assert "site" not in record.effective_config  # no site coordinates in the record
        rig.app.scheduler.step()
        assert len(rig.records("run")) == 1  # once only

    def test_without_a_camera_the_run_record_follows_after_the_wait(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path, config_extra="[services.core]\nrun_record_wait_s = 30.0\n")
        try:
            rig.camera.fail_open()  # the camera never opens
            rig.app.start()
            rig.app.tick()
            assert rig.records("run") == []
            rig.clock.advance(31.0)
            rig.app.tick()
            (record,) = rig.records("run")
            assert isinstance(record, RunRecord)
            assert "camera_model" not in record.versions
        finally:
            rig.app.stop()

    def test_the_start_is_an_event_and_the_endpoint_is_bound(self, rig: CoreRig) -> None:
        (started,) = rig.events("core.started")
        assert started.detail == {"instance": rig.app.instance}
        assert rig.app.bound_endpoint is not None
        with pytest.raises(RuntimeError, match="already started"):
            rig.app.start()

    def test_the_scheduler_gets_the_collaborators_and_the_handlers(self, rig: CoreRig) -> None:
        scheduler = rig.app.scheduler
        assert scheduler.state.value == "safe"
        assert {"burst", "replay", "sweep"} <= set(scheduler._handlers)
        assert scheduler._escalate is rig.app.escalator
        assert scheduler._alignment_sink == rig.app.alignment.sink


class TestHealth:
    def test_a_health_record_comes_at_once_and_then_every_interval(self, rig: CoreRig) -> None:
        rig.app.tick()
        assert len(rig.records("health")) == 1
        rig.clock.advance(30.0)
        rig.app.tick()
        assert len(rig.records("health")) == 1
        rig.clock.advance(31.0)
        rig.app.tick()
        assert len(rig.records("health")) == 2

    def test_the_record_holds_the_parts_that_core_knows(self, rig: CoreRig) -> None:
        rig.run_for(3.0)
        record = rig.records("health")[0]
        assert isinstance(record, HealthRecord)
        assert record.state in ("safe", "auto")  # the scheduler leaves safe within seconds
        assert record.components["core"] == "ok"
        assert record.components["acquire"] == "ok"
        assert record.queue_depth == 3
        assert record.free_space_gb is not None
        assert record.time_synchronized is True
        assert record.dark_due is True  # an empty dark library is due for a set
        assert "time_invalid" not in record.flags

    def test_an_unsynchronized_clock_flags_the_record(self, rig: CoreRig) -> None:
        rig.clock.set_status(ClockStatus(False, 86_400_000_000_000, "test"))
        rig.app.tick()
        assert "time_invalid" in rig.records("health")[-1].flags  # type: ignore[attr-defined]

    def test_the_first_open_of_the_camera_brings_a_record_at_once(self, rig: CoreRig) -> None:
        rig.app.tick()
        assert len(rig.records("health")) == 1  # the record of the start
        rig.clock.advance(5.0)
        rig.app.tick()
        assert len(rig.records("health")) == 1  # the interval has not passed
        rig.app.scheduler.step()  # the first step opens the camera
        rig.app.tick()
        assert len(rig.records("health")) == 2  # the open asked for a record with the camera

    def test_a_lost_acquire_is_a_failed_component(self, rig: CoreRig) -> None:
        rig.remote.health = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
            CameraTimeoutError("no answer")
        )
        rig.clock.advance(31.0)  # the start has passed, so a silent acquire is a failure
        rig.app.tick()
        record = rig.records("health")[-1]
        assert isinstance(record, HealthRecord)
        assert record.components["acquire"] == "failed"
        assert record.degraded is True


class TestTheRecordFollowsTheScheduler:
    """A change of state or of the camera brings a `health` record at once, not a minute later."""

    def test_a_failed_read_brings_a_record_that_says_why_within_seconds(
        self, tmp_path: Path
    ) -> None:
        # The camera shows the star. The search finds it with the bursts at 0 s and 15 s, and the
        # fast stream reads frames from about 16 s to the end of its period at 20 s.
        rig = build_rig(tmp_path, polaris=True)
        try:
            rig.run_for(17.0)
            before = rig.records("health")
            assert before[-1].components["camera"] == "ok"  # type: ignore[attr-defined]
            failed_at = rig.clock.utc_ns()
            rig.camera.fail_reads(*[CameraTimeoutError("no frame") for _ in range(3)])
            rig.run_for(10.0)
            records = rig.records("health")
            assert len(records) > len(before)
            record = records[-1]
            assert isinstance(record, HealthRecord)
            assert record.components["camera"] == "degraded"
            assert (record.quality or {})["components"] == (
                "camera: no frame arrived; the camera may be disconnected"
            )
            # The next scheduled record comes 60 seconds after the last one, so this one is early.
            assert (record.t_utc_ns - failed_at) / 1e9 < 20.0
        finally:
            rig.app.stop()

    def test_a_change_of_state_brings_a_record_at_once(self, rig: CoreRig) -> None:
        from seeingmon.scheduler import Pause

        rig.run_for(20.0)
        count = len(rig.records("health"))
        rig.app.scheduler.submit(Pause())
        rig.run_for(3.0)
        records = rig.records("health")
        assert len(records) > count
        assert records[-1].state == "paused"  # type: ignore[attr-defined]

    def test_a_quiet_run_writes_one_record_a_minute(self, rig: CoreRig) -> None:
        rig.run_for(10.0)
        count = len(rig.records("health"))
        rig.run_for(180.0)
        assert len(rig.records("health")) - count <= 4  # the scheduled ones, and nothing more


class TestEvents:
    def test_the_events_of_acquire_become_event_records_once(self, rig: CoreRig) -> None:
        rig.remote.log.append(
            HardwareEvent("warning", "camera.recovery", "Restarted the capture.", 1234, {"step": 1})
        )
        rig.app.tick()
        rig.clock.advance(6.0)
        rig.app.tick()
        found = events_of(rig.records("event"), "camera.recovery")
        assert [(e.message, e.t_utc_ns) for e in found] == [("Restarted the capture.", 1234)]

    def test_a_new_acquire_process_is_noticed(self, rig: CoreRig) -> None:
        rig.app.tick()
        rig.remote.instance = "acquire-two"
        rig.clock.advance(6.0)
        rig.app.tick()
        assert len(events_of(rig.records("event"), "acquire.restarted")) == 1

    def test_the_hardware_of_core_writes_events_through_the_same_writer(self, rig: CoreRig) -> None:
        rig.app.events(HardwareEvent("error", "heater.over_temperature", "Too warm.", 55, None))
        (found,) = events_of(rig.records("event"), "heater.over_temperature")
        assert found.provenance["source"] == "local"


class TestTheLadder:
    def test_a_camera_that_fails_for_good_climbs_to_a_restart_of_acquire(
        self, tmp_path: Path
    ) -> None:
        rig = build_rig(tmp_path, config_extra=SLOW_RETRY)
        try:
            rig.app.start()
            rig.camera.fail_reads(*[CameraTimeoutError("no frame") for _ in range(5000)])
            rig.camera.fail_recover()
            # The ladder climbs through the driver steps, two attempts each, with a backoff
            # between them. The scheduler then asks the supervisor to restart acquire.
            for _ in range(20_000):
                rig.app.scheduler.step()
                if rig.remote.restarts:
                    break
            assert rig.remote.restarts == ["recovery ladder: restart_acquire"]
            assert EscalationLevel.RESTART_ACQUIRE in rig.app.escalator.performed
            rig.app.scheduler.step()
            rig.app.tick()
            assert events_of(rig.records("event"), "escalation.restart_acquire")
        finally:
            rig.app.stop()


class TestShutdown:
    def test_the_stop_closes_the_camera_writes_the_last_event_and_closes_the_store(
        self, rig: CoreRig
    ) -> None:
        rig.run_for(3.0)
        path = rig.app.storage.layout.db_path  # type: ignore[union-attr]
        rig.app.stop("a test")
        assert ("close", None) in rig.camera.calls
        assert rig.app.storage is None
        with Store.open(path) as store:
            kinds = [r.kind for r in read_all(store, "event")]  # type: ignore[attr-defined]
        assert kinds[0] == "core.started"
        assert "core.stopped" in kinds
        rig.app.stop()  # twice is safe

    def test_a_replaced_part_is_stopped_with_the_rest(self, tmp_path: Path) -> None:
        class Heater:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def start(self) -> None:
                self.calls.append("start")

            def stop(self) -> None:
                self.calls.append("stop")

            def close(self) -> None:
                self.calls.append("close")

            def step(self) -> float:
                return 1.0

            def run(self, should_stop: Any) -> None:
                pass

            def status(self) -> Any:
                from types import SimpleNamespace

                return SimpleNamespace(state="off", enabled=True)

            def recent_duty(self, window_s: float) -> float | None:
                return 0.5

        heater = Heater()
        rig = build_rig(tmp_path, parts={"heater": heater})
        rig.app.start()
        rig.run_for(2.0)
        rig.app.stop()
        assert heater.calls[0] == "start"
        assert heater.calls[-1] == "close"
        assert "stop" in heater.calls


class TestBuildErrors:
    def test_the_two_analysis_windows_must_agree(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"analysis_window_s must equal fastpath\.window_s"):
            build_rig(tmp_path, analysis_window_s=15.0)

    def test_a_missing_catalog_is_a_clear_error_and_leaves_no_store_open(
        self, tmp_path: Path
    ) -> None:
        config = make_config(tmp_path)
        services = config.section("services", ServicesConfig)
        clock = VirtualClock(NIGHT)
        with pytest.raises(ConfigError, match="catalog_path is not set"):
            CoreApp(
                config,
                services,
                clock,
                ConnectionKey.from_text("a-test-key-of-more-than-32-characters"),
                parts=CoreParts(driver=FakeCameraDriver(clock), notifier=SystemdNotifier(env={})),
            )
        # The store opened and then closed again: a second open must not be refused.
        Store.open(tmp_path / "data" / "db" / "results.sqlite").close()


class TestThreads:
    def test_run_stops_on_request_and_exits_with_zero(self, tmp_path: Path) -> None:
        from seeingmon.clock import ScaledClock

        clock = ScaledClock(
            start_utc_ns=NIGHT, origin_real_ns=__import__("time").time_ns(), speed=20.0
        )
        rig = build_rig(
            tmp_path,
            threads=True,
            clock=clock,
            config_extra="[services.core]\nhealth_interval_s = 5.0\n",
        )
        outcome: list[int] = []
        thread = threading.Thread(target=lambda: outcome.append(rig.app.run()))
        thread.start()
        try:
            deadline = threading.Event()
            for _ in range(200):
                if len(rig.records("health")) >= 1 and rig.records("run"):
                    break
                deadline.wait(0.05)
            assert rig.records("run")
            assert rig.records("health")
        finally:
            rig.app.request_stop("a test")
            thread.join(30.0)
        assert outcome == [0]
        assert rig.app.exit_reason == "a test"

    def test_a_dead_scheduler_thread_ends_the_run_with_the_failure_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from seeingmon.clock import ScaledClock

        clock = ScaledClock(
            start_utc_ns=NIGHT, origin_real_ns=__import__("time").time_ns(), speed=20.0
        )
        rig = build_rig(tmp_path, threads=True, clock=clock)

        def explode(stop: threading.Event) -> None:
            raise RuntimeError("the scheduler broke")

        monkeypatch.setattr(rig.app.scheduler, "run", explode)
        assert rig.app.run() == EXIT_THREAD_DIED
        assert "core-scheduler died" in rig.app.exit_reason
        assert rig.app.storage is None
