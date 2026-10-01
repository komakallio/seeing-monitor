"""The small parts of `acquire`: the gate, the systemd notifier, the priority, and the factory."""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.drivers.base import CameraConfigError
from seeingmon.services.acquire.factory import create_camera_driver
from seeingmon.services.acquire.gate import DriverGate
from seeingmon.services.acquire.priority import raise_current_thread_priority
from seeingmon.services.notify import SystemdNotifier
from seeingmon.testing import FakeCameraDriver


class TestDriverGate:
    def test_a_read_and_a_control_call_never_overlap(self) -> None:
        gate = DriverGate()
        inside = 0
        worst = 0
        lock = threading.Lock()
        stop = threading.Event()

        def enter() -> None:
            nonlocal inside, worst
            with lock:
                inside += 1
                worst = max(worst, inside)
            time.sleep(0.0005)
            with lock:
                inside -= 1

        def reader() -> None:
            while not stop.is_set():
                with gate.read() as admitted:
                    if admitted:
                        enter()

        threads = [threading.Thread(target=reader) for _ in range(1)]
        for thread in threads:
            thread.start()
        try:
            for _ in range(40):
                with gate.control():
                    enter()
        finally:
            stop.set()
            for thread in threads:
                thread.join(5.0)
        assert worst == 1

    def test_a_control_call_gets_in_although_the_reader_loops_without_pause(self) -> None:
        gate = DriverGate()
        stop = threading.Event()
        reads = 0

        def reader() -> None:
            nonlocal reads
            while not stop.is_set():
                with gate.read() as admitted:
                    if admitted:
                        reads += 1

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            time.sleep(0.1)
            started = time.monotonic()
            for _ in range(20):
                with gate.control():
                    pass
            assert time.monotonic() - started < 30.0  # twenty calls, however busy the reader is
        finally:
            stop.set()
            thread.join(5.0)
        assert reads > 0

    def test_a_control_call_waits_for_the_read_in_progress(self) -> None:
        gate = DriverGate()
        order: list[str] = []
        reading = threading.Event()
        release = threading.Event()

        def reader() -> None:
            with gate.read():
                reading.set()
                release.wait(5.0)
                order.append("read done")

        thread = threading.Thread(target=reader)
        thread.start()
        assert reading.wait(5.0)
        timer = threading.Timer(0.2, release.set)
        timer.start()
        with gate.control():
            order.append("control in")
        thread.join(5.0)
        timer.join()
        assert order == ["read done", "control in"]

    def test_a_read_that_waits_can_be_aborted(self) -> None:
        gate = DriverGate()
        outcome: list[bool] = []
        abort = threading.Event()
        holding = threading.Event()
        release = threading.Event()

        def control() -> None:
            with gate.control():
                holding.set()
                release.wait(5.0)

        def read() -> None:
            with gate.read(abort.is_set) as admitted:
                outcome.append(admitted)

        holder = threading.Thread(target=control)
        holder.start()
        assert holding.wait(5.0)
        waiter = threading.Thread(target=read)
        waiter.start()
        time.sleep(0.15)
        abort.set()
        waiter.join(5.0)
        release.set()
        holder.join(5.0)
        assert outcome == [False]


class TestSystemdNotifier:
    def test_the_old_import_path_still_works(self) -> None:
        from seeingmon.services.acquire import notify as old_path

        assert old_path.SystemdNotifier is SystemdNotifier

    def test_the_status_goes_out_only_when_it_changes(self) -> None:
        sent: list[bytes] = []
        notifier = SystemdNotifier(env={}, send=sent.append)
        assert notifier.status_changed("running") is True
        assert notifier.status_changed("running") is False
        assert notifier.status_changed("degraded") is True
        assert sent == [b"STATUS=running", b"STATUS=degraded"]
        notifier.ready("running")  # READY=1 carries the text that it names as the last status
        assert notifier.status_changed("running") is False

    def collect(self, **env: str) -> tuple[SystemdNotifier, list[bytes]]:
        sent: list[bytes] = []
        return SystemdNotifier(env=env, send=sent.append, pid=1234), sent

    def test_it_does_nothing_without_a_socket(self) -> None:
        notifier = SystemdNotifier(env={})
        assert not notifier.enabled
        notifier.ready("up")
        notifier.watchdog()
        notifier.status("x")
        notifier.stopping()
        notifier.close()
        assert notifier.watchdog_interval_s is None

    def test_a_closed_notifier_sends_nothing(self) -> None:
        notifier, sent = self.collect()
        notifier.ready()
        notifier.close()
        notifier.watchdog()
        assert sent == [b"READY=1"]

    def test_messages_follow_the_protocol(self) -> None:
        notifier, sent = self.collect()
        notifier.ready("listening")
        notifier.watchdog()
        notifier.status("streaming\nsecond line")
        notifier.stopping()
        assert sent == [
            b"READY=1\nSTATUS=listening",
            b"WATCHDOG=1",
            b"STATUS=streaming second line",
            b"STOPPING=1",
        ]

    def test_ready_without_a_status_is_just_ready(self) -> None:
        notifier, sent = self.collect()
        notifier.ready()
        assert sent == [b"READY=1"]

    def test_the_heartbeat_interval_is_half_the_watchdog_time(self) -> None:
        notifier, _ = self.collect(WATCHDOG_USEC="10000000")
        assert notifier.watchdog_interval_s == pytest.approx(5.0)

    def test_a_watchdog_that_belongs_to_another_process_is_ignored(self) -> None:
        notifier, _ = self.collect(WATCHDOG_USEC="10000000", WATCHDOG_PID="999")
        assert notifier.watchdog_interval_s is None
        mine, _ = self.collect(WATCHDOG_USEC="10000000", WATCHDOG_PID="1234")
        assert mine.watchdog_interval_s == pytest.approx(5.0)

    @pytest.mark.parametrize("usec", ["", "0", "abc", "-5"])
    def test_a_bad_watchdog_time_means_no_watchdog(self, usec: str) -> None:
        notifier, _ = self.collect(WATCHDOG_USEC=usec)
        assert notifier.watchdog_interval_s is None

    def test_a_failing_send_is_logged_once_and_never_raised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls = 0

        def broken(data: bytes) -> None:
            nonlocal calls
            calls += 1
            raise ConnectionRefusedError("gone")

        notifier = SystemdNotifier(env={}, send=broken)
        with caplog.at_level(logging.WARNING):
            notifier.ready()
            notifier.watchdog()
            notifier.watchdog()
        assert calls == 1  # after the first failure the notifier stops trying
        assert sum("cannot notify systemd" in record.message for record in caplog.records) == 1

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix datagram sockets")
    def test_it_sends_datagrams_to_the_socket_that_systemd_names(self, short_dir: Path) -> None:
        path = str(short_dir / "notify.sock")
        receiver = socket.socket(socket.AddressFamily["AF_UNIX"], socket.SOCK_DGRAM)
        notifier = SystemdNotifier(env={"NOTIFY_SOCKET": path})
        try:
            receiver.bind(path)
            receiver.settimeout(5.0)
            assert notifier.enabled
            notifier.ready("up")
            notifier.watchdog()
            assert receiver.recv(1024) == b"READY=1\nSTATUS=up"
            assert receiver.recv(1024) == b"WATCHDOG=1"
        finally:
            notifier.close()
            receiver.close()

    @pytest.mark.skipif(sys.platform != "linux", reason="the abstract socket namespace")
    def test_an_abstract_socket_name_starts_with_an_at_sign(self) -> None:
        name = f"seeingmon-test-{os.getpid()}-{time.monotonic_ns()}"
        receiver = socket.socket(socket.AddressFamily["AF_UNIX"], socket.SOCK_DGRAM)
        notifier = SystemdNotifier(env={"NOTIFY_SOCKET": "@" + name})
        try:
            receiver.bind("\0" + name)
            receiver.settimeout(5.0)
            notifier.watchdog()
            assert receiver.recv(1024) == b"WATCHDOG=1"
        finally:
            notifier.close()
            receiver.close()

    @pytest.mark.skipif(sys.platform != "win32", reason="systemd runs on Linux only")
    def test_a_socket_on_windows_fails_softly(self) -> None:
        notifier = SystemdNotifier(env={"NOTIFY_SOCKET": "/run/example/notify"})
        notifier.ready()  # logs a warning, and never raises
        notifier.watchdog()


class TestPriority:
    def test_it_reports_what_happened_and_never_raises(self) -> None:
        result = raise_current_thread_priority()
        assert isinstance(result, str)
        assert result

    def test_it_runs_on_a_worker_thread_too(self) -> None:
        results: list[str] = []
        thread = threading.Thread(target=lambda: results.append(raise_current_thread_priority()))
        thread.start()
        thread.join(5.0)
        assert len(results) == 1

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux scheduling calls")
    def test_without_the_privilege_the_thread_keeps_its_priority(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: object, **kwargs: object) -> None:
            raise PermissionError("not permitted")

        monkeypatch.setattr(os, "sched_setscheduler", refuse)
        monkeypatch.setattr(os, "setpriority", refuse)
        assert "not permitted" in raise_current_thread_priority()

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux scheduling calls")
    def test_a_nice_value_is_the_second_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(*args: object, **kwargs: object) -> None:
            raise PermissionError("not permitted")

        calls: list[tuple[object, ...]] = []
        monkeypatch.setattr(os, "sched_setscheduler", refuse)
        monkeypatch.setattr(os, "setpriority", lambda *args: calls.append(args))
        assert raise_current_thread_priority() == "nice -10"
        assert calls[0][2] == -10


class TestFactory:
    def test_the_fake_driver_takes_its_options(self) -> None:
        clock = VirtualClock()
        driver = create_camera_driver(
            "fake",
            profile=None,
            clock=clock,
            options={"adc_bits": 12, "temperature_c": 7, "row_time_s": 4e-5},
        )
        assert isinstance(driver, FakeCameraDriver)
        driver.open()
        assert driver.read_temperature_c() == 7.0

    def test_an_option_that_the_fake_does_not_know_is_a_config_error(self) -> None:
        with pytest.raises(CameraConfigError, match="no option named 'colour'"):
            create_camera_driver("fake", profile=None, clock=VirtualClock(), options={"colour": 1})

    @pytest.mark.parametrize("value", ["12", True, None, [1]])
    def test_an_option_that_is_not_a_number_is_a_config_error(self, value: object) -> None:
        with pytest.raises(CameraConfigError, match="must be a number"):
            create_camera_driver(
                "fake", profile=None, clock=VirtualClock(), options={"adc_bits": value}
            )

    def test_other_names_go_through_the_driver_factory(self) -> None:
        driver = create_camera_driver("sim", profile=None, clock=VirtualClock(), options={})
        assert driver.name == "sim"

    def test_an_unknown_driver_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown driver"):
            create_camera_driver("nope", profile=None, clock=VirtualClock(), options={})


class TestServiceHelpers:
    def test_the_default_guard_is_the_call_watchdog_of_the_hardware_lane(self) -> None:
        from seeingmon.hardware.asi.watchdog import CallWatchdog
        from seeingmon.services.acquire.service import default_guard

        guard = default_guard(VirtualClock())
        assert isinstance(guard, CallWatchdog)
        with guard.guard("a quick call", 5.0):
            pass
        assert guard.hang_count == 0

    def test_a_fatal_error_ends_the_process_with_its_own_exit_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from seeingmon.services.acquire.service import EXIT_THREAD_DIED, exit_on_fatal

        codes: list[int] = []
        monkeypatch.setattr(os, "_exit", codes.append)
        exit_on_fatal("a service thread died")
        assert codes == [EXIT_THREAD_DIED]
        assert "a service thread died" in capsys.readouterr().err

    def test_every_timing_setting_reaches_the_stamper(self) -> None:
        from seeingmon.services.acquire.service import timing_config
        from seeingmon.services.config import AcquireSettings

        settings = AcquireSettings(
            fit_window=64,
            fit_warmup=10,
            latency_s=0.004,
            latency_sigma_s=0.002,
            arrival_jitter_s=0.001,
            unknown_clock_error_s=0.25,
            invalid_clock_error_s=7.0,
            outlier_sigmas=5.0,
            outlier_floor_s=0.01,
            step_frames=4,
        )
        config = timing_config(settings)
        assert (config.window, config.warmup, config.step_frames) == (64, 10, 4)
        assert (config.latency_s, config.latency_sigma_s) == (0.004, 0.002)
        assert (config.arrival_jitter_s, config.unknown_clock_error_s) == (0.001, 0.25)
        assert (config.invalid_clock_error_s, config.outlier_sigmas) == (7.0, 5.0)
        assert config.outlier_floor_s == 0.01

    def test_the_health_summary_is_json_and_one_line(self) -> None:
        import json
        from dataclasses import fields

        from seeingmon.services.acquire.health import AcquireHealth

        values: dict[str, object] = {}
        for field in fields(AcquireHealth):
            kind = str(field.type)
            if "bool" in kind:
                values[field.name] = False
            elif "int" in kind and "None" not in kind:
                values[field.name] = 1
            elif "float" in kind:
                values[field.name] = 2.5
            elif "None" in kind:
                values[field.name] = None
            else:
                values[field.name] = "text"
        values.update(state="streaming", capturing=True, frame_rate_hz=88.5, last_error="boom")
        health = AcquireHealth(**values)  # type: ignore[arg-type]
        assert json.loads(json.dumps(health.to_json()))["state"] == "streaming"
        summary = health.summary()
        assert "\n" not in summary
        assert summary.startswith("streaming, 88.5 fps")
        assert summary.endswith("last error: boom")

    def test_the_remote_driver_is_built_from_the_configuration(self, tmp_path: Path) -> None:
        from seeingmon.config import load_config
        from seeingmon.services.config import ServicesConfig
        from seeingmon.services.remote import RemoteCameraDriver

        key = "a-test-key-with-32-characters-long"
        config = load_config(
            local_file=tmp_path / "none.toml",
            env={
                "SEEINGMON_SERVICES__CONNECTION_KEY": key,
                "SEEINGMON_SERVICES__ACQUIRE_ADDRESS": '"seeingmon-test-acquire"',
                "SEEINGMON_SERVICES__RPC_TIMEOUT_S": "7",
            },
        )
        services = config.section("services", ServicesConfig)
        driver = RemoteCameraDriver.from_config(services, env={})
        assert driver.name == "remote"
        assert not driver.connected
        assert driver._rpc_timeout_s == 7.0
        assert driver._window.messages == services.stream_window_messages
