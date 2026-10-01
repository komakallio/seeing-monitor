"""The power-cycle hook: routes, rate limits, the state file, and what the events never carry."""

from __future__ import annotations

import json
import secrets
import socket
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from seeingmon.clock import NS_PER_S, ClockStatus, VirtualClock
from seeingmon.hardware.events import HardwareEvent
from seeingmon.hardware.power import (
    CommandResult,
    CommandRoute,
    HttpResult,
    HttpRoute,
    PowerConfig,
    PowerCycle,
    PowerCycleError,
    PowerOutcome,
    SubprocessRunner,
    UrllibSender,
    expand,
)

SECRET = secrets.token_hex(8)  # a fresh value each run, so no literal looks like a credential


class FakeRunner:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.result = result or CommandResult(0)
        self.calls: list[tuple[list[str], float]] = []
        self.during: list[object] = []  # what a callback saw while the command ran
        self.on_run: Any = None

    def run(self, argv: Sequence[str], timeout_s: float) -> CommandResult:
        self.calls.append((list(argv), timeout_s))
        if self.on_run is not None:
            self.during.append(self.on_run())
        return self.result


class FakeSender:
    def __init__(self, result: HttpResult | None = None) -> None:
        self.result = result or HttpResult(200)
        self.calls: list[tuple[str, str, dict[str, str], bytes | None, float, bool]] = []

    def send(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        verify_tls: bool,
    ) -> HttpResult:
        self.calls.append((method, url, dict(headers), body, timeout_s, verify_tls))
        return self.result


@dataclass
class Rig:
    clock: VirtualClock
    power: PowerCycle
    runner: FakeRunner
    sender: FakeSender
    events: list[HardwareEvent] = field(default_factory=list)

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]


def command_config(**overrides: Any) -> PowerConfig:
    values: dict[str, Any] = {
        "route": "command",
        "command": CommandRoute(argv=["power-cycle-program", "--target", "${PLUG_NAME}"]),
        **overrides,
    }
    return PowerConfig(**values)


def http_config(**overrides: Any) -> PowerConfig:
    values: dict[str, Any] = {
        "route": "http",
        "http": HttpRoute(
            method="POST",
            url="http://${PLUG_HOST}/relay",
            headers={"Authorization": "Bearer ${PLUG_TOKEN}", "Content-Type": "text/plain"},
            body="cycle ${PLUG_NAME}",
        ),
        **overrides,
    }
    return PowerConfig(**values)


ENV = {"PLUG_NAME": "plug-1", "PLUG_HOST": "example.com", "PLUG_TOKEN": SECRET}


def make(config: PowerConfig, *, env: Mapping[str, str] | None = None) -> Rig:
    clock = VirtualClock()
    runner, sender = FakeRunner(), FakeSender()
    events: list[HardwareEvent] = []
    power = PowerCycle(
        config,
        clock=clock,
        runner=runner,
        sender=sender,
        on_event=events.append,
        env=ENV if env is None else env,
    )
    return Rig(clock, power, runner, sender, events)


class TestConfig:
    def test_the_default_route_is_none(self) -> None:
        config = PowerConfig()
        assert (config.route, config.dry_run, config.max_per_day) == ("none", False, 3)
        assert config.min_interval_s == 3600.0

    @pytest.mark.parametrize(
        "values",
        [
            {"route": "command"},  # no argv
            {"route": "http"},  # no url
            {"route": "http", "http": HttpRoute(url="ftp://example.com/x")},
            {"route": "shell"},
            {"max_per_day": 0},
            {"min_interval_s": -1},
            {"timeout_s": 0},
            {"surprise": 1},
        ],
    )
    def test_a_bad_config_is_refused(self, values: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            PowerConfig(**values)

    def test_a_url_can_start_with_a_variable(self) -> None:
        PowerConfig(route="http", http=HttpRoute(url="${PLUG_URL}"))


class TestExpand:
    def test_replaces_every_variable_and_leaves_other_text(self) -> None:
        assert expand("a ${X} b ${Y} $Z ${}", {"X": "1", "Y": "2"}) == "a 1 b 2 $Z ${}"

    def test_a_missing_variable_is_named_and_no_value_is_shown(self) -> None:
        with pytest.raises(PowerCycleError, match="PLUG_TOKEN is not set") as raised:
            expand("Bearer ${PLUG_TOKEN}", {"OTHER": SECRET})
        assert SECRET not in str(raised.value)


class TestNoRoute:
    def test_nothing_runs_and_the_caller_learns_why(self) -> None:
        rig = make(PowerConfig())
        result = rig.power.request("health failed")
        assert result.outcome is PowerOutcome.UNAVAILABLE
        assert rig.kinds() == ["power.cycle_unavailable"]
        assert rig.runner.calls == []
        assert rig.sender.calls == []


class TestCommandRoute:
    def test_runs_the_expanded_argument_list_without_a_shell(self) -> None:
        rig = make(command_config(timeout_s=12.0))
        result = rig.power.request("camera stalled for 10 minutes")
        assert result.outcome is PowerOutcome.DONE
        assert rig.runner.calls == [(["power-cycle-program", "--target", "plug-1"], 12.0)]
        assert rig.kinds() == ["power.cycle_requested", "power.cycle_done"]
        assert rig.events[0].level == "warning"
        assert rig.events[0].detail == {
            "route": "command",
            "reason": "camera stalled for 10 minutes",
            "attempts_today": 1,
        }

    def test_a_failing_exit_code_is_a_failure(self) -> None:
        rig = make(command_config())
        rig.runner.result = CommandResult(3)
        result = rig.power.request("test")
        assert (result.outcome, result.detail) == (PowerOutcome.FAILED, "exit code 3")
        assert rig.kinds() == ["power.cycle_requested", "power.cycle_failed"]
        assert rig.events[1].level == "error"

    def test_a_timeout_is_a_failure(self) -> None:
        rig = make(command_config())
        rig.runner.result = CommandResult(None, timed_out=True)
        assert rig.power.request("test").detail == "timeout"

    def test_no_event_carries_a_command_a_url_or_a_secret(self) -> None:
        rig = make(command_config())
        rig.power.request("test")
        text = repr(rig.events)
        assert "power-cycle-program" not in text
        assert "plug-1" not in text

    def test_a_missing_variable_fails_before_anything_runs_and_costs_no_attempt(self) -> None:
        rig = make(command_config(), env={})
        result = rig.power.request("test")
        assert result.outcome is PowerOutcome.FAILED
        assert "PLUG_NAME" in result.detail
        assert rig.runner.calls == []
        assert rig.power.attempts_today() == 0


class TestSubprocessRunner:
    def test_arguments_reach_the_program_untouched(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker"
        target = tmp_path / "out.txt"
        awkward = f"a b; echo injected > {marker}"
        script = "import sys; open(sys.argv[1], 'w').write(sys.argv[2])"
        result = SubprocessRunner().run([sys.executable, "-c", script, str(target), awkward], 30.0)
        assert result == CommandResult(0)
        assert target.read_text() == awkward
        assert not marker.exists()  # a shell would have run the second command

    def test_the_exit_code_comes_back(self) -> None:
        result = SubprocessRunner().run([sys.executable, "-c", "import sys; sys.exit(3)"], 30.0)
        assert result == CommandResult(3)

    def test_a_command_that_runs_too_long_times_out(self) -> None:
        started = time.monotonic()
        result = SubprocessRunner().run([sys.executable, "-c", "import time; time.sleep(60)"], 0.3)
        assert result == CommandResult(None, timed_out=True)
        assert time.monotonic() - started < 30

    def test_a_missing_program_is_a_failure_that_the_hook_reports(self) -> None:
        events: list[HardwareEvent] = []
        power = PowerCycle(
            PowerConfig(route="command", command=CommandRoute(argv=["no-such-program-xyz"])),
            clock=VirtualClock(),
            on_event=events.append,
            env={},
        )
        result = power.request("test")
        assert (result.outcome, result.detail) == (PowerOutcome.FAILED, "FileNotFoundError")
        assert "no-such-program-xyz" not in repr(events)


class Recorder(BaseHTTPRequestHandler):
    server: Any

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append((self.command, self.path, dict(self.headers), body))
        time.sleep(self.server.delay)
        self.send_response(self.server.status)
        self.end_headers()

    # The server calls the method that is named after the HTTP verb.
    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _handle  # noqa: N815

    def log_message(self, format: str, *args: Any) -> None:
        return None


@dataclass
class Plug:
    server: ThreadingHTTPServer

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/relay"

    @property
    def requests(self) -> list[tuple[str, str, dict[str, str], bytes]]:
        return self.server.requests  # type: ignore[attr-defined, no-any-return]


@pytest.fixture
def plug() -> Iterator[Plug]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    server.requests = []  # type: ignore[attr-defined]
    server.status = 200  # type: ignore[attr-defined]
    server.delay = 0.0  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield Plug(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10.0)


def real_http(
    config: PowerConfig, env: Mapping[str, str]
) -> tuple[PowerCycle, list[HardwareEvent]]:
    events: list[HardwareEvent] = []
    power = PowerCycle(
        config, clock=VirtualClock(), sender=UrllibSender(), on_event=events.append, env=env
    )
    return power, events


class TestHttpRoute:
    def test_sends_the_request_with_the_expanded_header_and_body(self) -> None:
        rig = make(http_config(timeout_s=7.0))
        result = rig.power.request("test")
        assert result.outcome is PowerOutcome.DONE
        method, url, headers, body, timeout, verify = rig.sender.calls[0]
        assert (method, url, timeout, verify) == ("POST", "http://example.com/relay", 7.0, True)
        assert headers == {"Authorization": f"Bearer {SECRET}", "Content-Type": "text/plain"}
        assert body == b"cycle plug-1"

    def test_no_event_carries_the_url_or_the_token(self) -> None:
        rig = make(http_config())
        rig.power.request("test")
        text = repr(rig.events)
        assert SECRET not in text
        assert "example.com" not in text
        assert rig.events[1].detail == {"route": "http", "status": 200}

    def test_a_non_success_status_is_a_failure(self) -> None:
        rig = make(http_config())
        rig.sender.result = HttpResult(503)
        result = rig.power.request("test")
        assert (result.outcome, result.detail) == (PowerOutcome.FAILED, "status 503")

    def test_a_transport_error_is_a_failure_named_by_its_class(self) -> None:
        rig = make(http_config())
        rig.sender.result = HttpResult(None, "ConnectionRefusedError")
        assert rig.power.request("test").detail == "ConnectionRefusedError"

    def test_a_url_that_leaves_http_after_the_expansion_is_refused(self) -> None:
        config = PowerConfig(route="http", http=HttpRoute(url="${PLUG_URL}"))
        rig = make(config, env={"PLUG_URL": "file:///etc/hostname"})
        result = rig.power.request("test")
        assert result.outcome is PowerOutcome.FAILED
        assert rig.sender.calls == []

    def test_the_tls_option_reaches_the_sender(self) -> None:
        rig = make(http_config(http=HttpRoute(url="https://example.com/x", verify_tls=False)))
        rig.power.request("test")
        assert rig.sender.calls[0][5] is False

    def test_a_get_request_has_no_body(self) -> None:
        config = http_config(http=HttpRoute(method="GET", url="http://example.com/cycle"))
        rig = make(config)
        rig.power.request("test")
        assert rig.sender.calls[0][0] == "GET"
        assert rig.sender.calls[0][3] is None


class TestRealHttp:
    """The hook against a local server, through `urllib`."""

    def config(self, plug: Plug, **overrides: Any) -> PowerConfig:
        route = HttpRoute(
            method="POST",
            url="${PLUG_URL}",
            headers={"Authorization": "Bearer ${PLUG_TOKEN}"},
            body="cycle",
        )
        return PowerConfig(route="http", http=route, timeout_s=5.0, **overrides)

    def env(self, plug: Plug) -> dict[str, str]:
        return {"PLUG_URL": plug.url, "PLUG_TOKEN": SECRET}

    def test_a_successful_cycle(self, plug: Plug) -> None:
        power, events = real_http(self.config(plug), self.env(plug))
        assert power.request("test").outcome is PowerOutcome.DONE
        method, path, headers, body = plug.requests[0]
        assert (method, path, body) == ("POST", "/relay", b"cycle")
        assert headers["Authorization"] == f"Bearer {SECRET}"
        assert [event.kind for event in events] == ["power.cycle_requested", "power.cycle_done"]
        assert SECRET not in repr(events)

    def test_an_error_status_is_a_failure(self, plug: Plug) -> None:
        plug.server.status = 500  # type: ignore[attr-defined]
        power, _ = real_http(self.config(plug), self.env(plug))
        result = power.request("test")
        assert (result.outcome, result.detail) == (PowerOutcome.FAILED, "status 500")

    def test_a_slow_server_times_out(self, plug: Plug) -> None:
        plug.server.delay = 1.5  # type: ignore[attr-defined]
        config = PowerConfig(
            route="http", http=HttpRoute(url="${PLUG_URL}", method="GET"), timeout_s=0.2
        )
        power, _ = real_http(config, self.env(plug))
        result = power.request("test")
        assert result.outcome is PowerOutcome.FAILED
        assert result.detail in ("TimeoutError", "timeout")

    def test_a_server_that_is_down_is_a_failure_without_an_address(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        config = PowerConfig(route="http", http=HttpRoute(url="${PLUG_URL}"), timeout_s=2.0)
        power, events = real_http(config, {"PLUG_URL": f"http://127.0.0.1:{port}/relay"})
        result = power.request("test")
        assert result.outcome is PowerOutcome.FAILED
        assert str(port) not in repr(events)
        assert str(port) not in result.detail


class TestDryRun:
    def test_reports_and_sends_nothing(self) -> None:
        rig = make(command_config(dry_run=True))
        result = rig.power.request("rehearsal")
        assert result.outcome is PowerOutcome.DRY_RUN
        assert rig.runner.calls == []
        assert rig.kinds() == ["power.cycle_dry_run"]
        assert rig.events[0].level == "info"

    def test_a_dry_run_still_checks_the_environment(self) -> None:
        rig = make(http_config(dry_run=True), env={})
        assert rig.power.request("rehearsal").outcome is PowerOutcome.FAILED

    def test_a_dry_run_never_counts_against_the_limits(self) -> None:
        rig = make(command_config(dry_run=True, max_per_day=1))
        for _ in range(5):
            assert rig.power.request("rehearsal").outcome is PowerOutcome.DRY_RUN
        assert rig.power.attempts_today() == 0


class TestRateLimits:
    def test_the_minimum_interval_blocks_a_quick_second_attempt(self) -> None:
        rig = make(command_config(min_interval_s=3600.0))
        assert rig.power.request("first").outcome is PowerOutcome.DONE
        rig.clock.advance(1800.0)
        blocked = rig.power.request("second")
        assert (blocked.outcome, blocked.detail) == (PowerOutcome.RATE_LIMITED, "minimum_interval")
        assert len(rig.runner.calls) == 1
        assert rig.events[-1].kind == "power.cycle_blocked"
        assert rig.events[-1].detail == {
            "route": "command",
            "limit": "minimum_interval",
            "reason": "second",
        }
        rig.clock.advance(1800.0)
        assert rig.power.request("third").outcome is PowerOutcome.DONE

    def test_the_daily_maximum_caps_the_attempts_in_any_24_hours(self) -> None:
        rig = make(command_config(min_interval_s=3600.0, max_per_day=3))
        for _ in range(3):
            assert rig.power.request("a").outcome is PowerOutcome.DONE
            rig.clock.advance(3600.0)
        blocked = rig.power.request("fourth")
        assert (blocked.outcome, blocked.detail) == (PowerOutcome.RATE_LIMITED, "daily_maximum")
        rig.clock.advance(21 * 3600.0 - 1)  # 24 hours after the first attempt, less a second
        assert rig.power.request("still blocked").outcome is PowerOutcome.RATE_LIMITED
        rig.clock.advance(2.0)
        assert rig.power.request("after a day").outcome is PowerOutcome.DONE

    def test_a_failed_attempt_counts(self) -> None:
        rig = make(command_config(max_per_day=1, min_interval_s=0.0))
        rig.runner.result = CommandResult(1)
        assert rig.power.request("a").outcome is PowerOutcome.FAILED
        assert rig.power.request("b").outcome is PowerOutcome.RATE_LIMITED

    def test_a_zero_interval_allows_back_to_back_attempts_up_to_the_daily_cap(self) -> None:
        rig = make(command_config(max_per_day=2, min_interval_s=0.0))
        assert rig.power.request("a").outcome is PowerOutcome.DONE
        assert rig.power.request("b").outcome is PowerOutcome.DONE
        assert rig.power.request("c").outcome is PowerOutcome.RATE_LIMITED

    def test_an_unsynchronized_clock_blocks_when_an_attempt_is_on_record(self) -> None:
        rig = make(command_config(min_interval_s=0.0))
        rig.clock.set_status(ClockStatus(synchronized=False, error_bound_ns=None, source="t"))
        assert rig.power.request("first").outcome is PowerOutcome.DONE  # nothing on record yet
        blocked = rig.power.request("second")
        assert (blocked.outcome, blocked.detail) == (
            PowerOutcome.RATE_LIMITED,
            "clock_not_synchronized",
        )

    def test_a_clock_that_stepped_back_behind_the_last_attempt_blocks(self) -> None:
        rig = make(command_config(min_interval_s=0.0))
        rig.power.request("first")
        rig.clock.step_utc_ns(-3600 * NS_PER_S)
        blocked = rig.power.request("second")
        assert blocked.detail == "clock_behind_the_last_attempt"

    def test_the_reason_is_cut_to_200_characters(self) -> None:
        rig = make(command_config())
        rig.power.request("x" * 500)
        assert rig.events[0].detail is not None
        assert len(rig.events[0].detail["reason"]) == 200


class TestStateFile:
    def test_a_new_instance_remembers_the_attempts_of_the_last_one(self, tmp_path: Path) -> None:
        state = tmp_path / "power.json"
        first = make(command_config(state_file=str(state)))
        assert first.power.request("a").outcome is PowerOutcome.DONE
        saved = json.loads(state.read_text())
        assert saved == {"attempts_utc_ns": [first.clock.utc_ns()]}
        # The reboot: a new process, a clock that has moved a little.
        second = make(command_config(state_file=str(state)))
        second.clock.advance_to_utc_ns(first.clock.utc_ns() + 60 * NS_PER_S)
        blocked = second.power.request("b")
        assert (blocked.outcome, blocked.detail) == (PowerOutcome.RATE_LIMITED, "minimum_interval")

    def test_the_attempt_is_on_disk_before_the_command_runs(self, tmp_path: Path) -> None:
        state = tmp_path / "power.json"
        rig = make(command_config(state_file=str(state)))
        rig.runner.on_run = lambda: json.loads(state.read_text())["attempts_utc_ns"]
        rig.power.request("a")
        assert rig.runner.during == [[rig.clock.utc_ns()]]

    def test_old_attempts_leave_the_file(self, tmp_path: Path) -> None:
        state = tmp_path / "power.json"
        rig = make(command_config(state_file=str(state), min_interval_s=0.0))
        rig.power.request("a")
        rig.clock.advance(2 * 24 * 3600.0)
        rig.power.request("b")
        assert len(json.loads(state.read_text())["attempts_utc_ns"]) == 1

    @pytest.mark.parametrize(
        "content", ["not json", '{"attempts_utc_ns": "x"}', "[1, 2]", '{"attempts_utc_ns": [true]}']
    )
    def test_a_damaged_file_starts_empty_and_reports(self, tmp_path: Path, content: str) -> None:
        state = tmp_path / "power.json"
        state.write_text(content)
        rig = make(command_config(state_file=str(state)))
        assert rig.kinds() == ["power.state_unreadable"]
        assert rig.events[0].level == "warning"
        assert rig.power.request("a").outcome is PowerOutcome.DONE

    def test_a_state_file_that_cannot_be_written_warns_and_the_cycle_still_runs(
        self, tmp_path: Path
    ) -> None:
        blocked = tmp_path / "power.json"
        blocked.mkdir()  # a directory where the file should be
        rig = make(command_config(state_file=str(blocked)))
        result = rig.power.request("a")
        assert result.outcome is PowerOutcome.DONE
        assert "power.state_unwritable" in rig.kinds()
        assert len(rig.runner.calls) == 1
