"""The runner: one socket for each address, one event loop, a clean stop, and systemd."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from seeingmon.services.acquire.notify import SystemdNotifier
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.core_client import CoreUnavailableError, FakeCoreClient
from seeingmon.services.web.runner import (
    EXIT_STARTUP_FAILURE,
    BindError,
    WebRunner,
    bind_sockets,
    format_address,
    probe_http,
    url_for,
)
from tests.services.conftest import wait_until
from tests.services.web.server import (
    API,
    IPV6_LOOPBACK,
    LOOPBACK,
    OTHER_LOOPBACK,
    Running,
    assert_nobody_listens,
    fetch,
    free_port,
    handshake,
    needs_ipv6,
    needs_second_loopback,
    needs_websocket_library,
)

# --- The sockets -----------------------------------------------------------------------------


def test_an_address_gives_a_url_and_a_text_with_the_port() -> None:
    assert format_address("192.0.2.5", 8080) == "192.0.2.5:8080"
    assert format_address("2001:db8::5", 8080) == "[2001:db8::5]:8080"
    assert url_for("192.0.2.5", 8080) == "http://192.0.2.5:8080/"
    assert url_for("2001:db8::5", 80) == "http://[2001:db8::5]:80/"


def test_one_socket_listens_for_the_address() -> None:
    sockets = bind_sockets([LOOPBACK], 0)
    try:
        assert len(sockets) == 1
        assert sockets[0].getsockname()[0] == LOOPBACK
        assert sockets[0].getsockname()[1] > 0
    finally:
        for sock in sockets:
            sock.close()


@needs_second_loopback
def test_port_zero_gives_every_socket_the_port_that_the_first_one_got() -> None:
    sockets = bind_sockets([LOOPBACK, OTHER_LOOPBACK], 0)
    try:
        assert [sock.getsockname()[0] for sock in sockets] == [LOOPBACK, OTHER_LOOPBACK]
        assert len({sock.getsockname()[1] for sock in sockets}) == 1
    finally:
        for sock in sockets:
            sock.close()


@needs_second_loopback
def test_a_fixed_port_is_shared_by_every_address() -> None:
    port = free_port()
    sockets = bind_sockets([LOOPBACK, OTHER_LOOPBACK], port)
    try:
        assert {sock.getsockname()[1] for sock in sockets} == {port}
    finally:
        for sock in sockets:
            sock.close()


@needs_second_loopback
def test_a_failed_bind_names_the_failed_address_and_releases_the_others() -> None:
    (blocker,) = bind_sockets([OTHER_LOOPBACK], 0)
    port = int(blocker.getsockname()[1])
    try:
        with pytest.raises(BindError) as raised:
            bind_sockets([LOOPBACK, OTHER_LOOPBACK], port)
        message = str(raised.value)
        assert f"{OTHER_LOOPBACK}:{port}" in message
        assert f"{LOOPBACK}:" not in message
        again = bind_sockets([LOOPBACK], port)  # the first socket was closed
        for sock in again:
            sock.close()
    finally:
        blocker.close()


def test_an_address_of_another_machine_cannot_be_bound() -> None:
    with pytest.raises(BindError) as raised:
        bind_sockets(["192.0.2.1"], 8080)
    assert "192.0.2.1:8080" in str(raised.value)


def test_text_that_is_not_an_address_cannot_be_bound() -> None:
    with pytest.raises(BindError) as raised:
        bind_sockets(["not-an-address"], 8080)
    assert "not-an-address:8080" in str(raised.value)


@needs_ipv6
def test_an_ipv6_socket_accepts_ipv6_only() -> None:
    (sock,) = bind_sockets([IPV6_LOOPBACK], 0)
    try:
        assert sock.family == socket.AF_INET6
        assert sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
    finally:
        sock.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the exclusive bind is a Windows rule")
def test_on_windows_no_other_socket_can_share_the_port() -> None:
    (first,) = bind_sockets([LOOPBACK], 0)
    other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(PermissionError):
            other.bind((LOOPBACK, int(first.getsockname()[1])))
    finally:
        other.close()
        first.close()


# --- A running server ------------------------------------------------------------------------


@pytest.fixture
def start(make_app: Callable[..., FastAPI], settings: WebSettings) -> Callable[..., Running]:
    """Build a `Running` server over the app of the tests, on a free port.

    The keyword arguments are `[web]` values. The app gets the same settings as the runner, as
    `seeingmon web` does, so the host rule knows the addresses that the runner listens on.
    Pass `wrap` to put an ASGI wrapper around the app, and `notifier` and the other runner options
    by name.
    """

    def build(
        *,
        wrap: Callable[[ASGIApp], ASGIApp] | None = None,
        app_options: dict[str, Any] | None = None,
        **values: Any,
    ) -> Running:
        runner_options = {
            name: values.pop(name)
            for name in ("notifier", "probe_timeout_s", "watchdog_interval_s", "on_started")
            if name in values
        }
        chosen = WebSettings.model_validate(
            {**settings.model_dump(), "port": 0, "shutdown_timeout_s": 0.5, **values}
        )
        app = make_app(settings=chosen, **(app_options or {}))
        return Running(wrap(app) if wrap is not None else app, chosen, **runner_options)

    return build


def test_the_server_answers_on_its_address_and_stops_cleanly(
    start: Callable[..., Running],
) -> None:
    with start() as server:
        status, headers, body = fetch(LOOPBACK, server.port, f"{API}/status")
        assert status == 200
        assert headers["content-type"] == "application/json"
        assert json.loads(body)["core"]["reachable"] is True
        page, _, html = fetch(LOOPBACK, server.port, "/")
        assert page == 200
        assert b"<html" in html.lower()
    assert server.exit_code == 0
    assert server.error is None
    assert_nobody_listens(LOOPBACK, server.port)  # the socket is closed


def test_the_server_sends_no_server_header(start: Callable[..., Running]) -> None:
    with start() as server:
        _status, headers, _body = fetch(LOOPBACK, server.port, f"{API}/status")
    assert "server" not in headers


class SpyApp:
    """An ASGI app that forwards to another one and records what it saw."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.loops: set[int] = set()
        self.servers: set[tuple[str, int]] = set()
        self.lifespan: list[str] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":

            async def spy_send(message: Message) -> None:
                self.lifespan.append(message["type"])
                await send(message)

            await self.app(scope, receive, spy_send)
            return
        self.loops.add(id(asyncio.get_running_loop()))
        server = scope.get("server")
        if server:
            self.servers.add((str(server[0]), int(server[1])))
        await self.app(scope, receive, send)


@needs_second_loopback
def test_one_event_loop_serves_two_listening_addresses(start: Callable[..., Running]) -> None:
    spies: list[SpyApp] = []

    def wrap(app: ASGIApp) -> ASGIApp:
        spies.append(SpyApp(app))
        return spies[0]

    with start(wrap=wrap, bind_address=LOOPBACK, extra_bind_addresses=[OTHER_LOOPBACK]) as server:
        assert server.runner.addresses == (LOOPBACK, OTHER_LOOPBACK)
        for address in (LOOPBACK, OTHER_LOOPBACK):
            assert fetch(address, server.port, f"{API}/status")[0] == 200
            assert fetch(address, server.port, "/")[0] == 200
    (spy,) = spies
    assert len(spy.loops) == 1
    assert {address for address, _port in spy.servers} == {LOOPBACK, OTHER_LOOPBACK}
    assert {port for _address, port in spy.servers} == {server.port}
    assert server.exit_code == 0


@needs_second_loopback
def test_the_server_listens_on_the_named_addresses_and_no_other(
    start: Callable[..., Running],
) -> None:
    with start(bind_address=LOOPBACK) as server:
        assert fetch(LOOPBACK, server.port, f"{API}/status")[0] == 200
        assert_nobody_listens(OTHER_LOOPBACK, server.port)


@needs_second_loopback
def test_a_repeated_address_opens_one_socket(start: Callable[..., Running]) -> None:
    with start(
        bind_address=LOOPBACK, extra_bind_addresses=["localhost", LOOPBACK, OTHER_LOOPBACK]
    ) as server:
        assert server.runner.addresses == (LOOPBACK, OTHER_LOOPBACK)


@needs_ipv6
def test_an_ipv4_and_an_ipv6_address_listen_side_by_side(start: Callable[..., Running]) -> None:
    with start(bind_address=LOOPBACK, extra_bind_addresses=[IPV6_LOOPBACK]) as server:
        assert fetch(LOOPBACK, server.port, f"{API}/status")[0] == 200
        assert fetch(IPV6_LOOPBACK, server.port, f"{API}/status")[0] == 200
        assert server.runner.url == f"http://{LOOPBACK}:{server.port}/"


def test_a_host_that_is_not_allowed_gets_400_over_a_real_connection(
    start: Callable[..., Running],
) -> None:
    with start() as server:
        status, _headers, body = fetch(LOOPBACK, server.port, f"{API}/status", host="evil.example")
        assert status == 400
        assert json.loads(body)["error"]["code"] == "host_not_allowed"
        assert fetch(LOOPBACK, server.port, "/", host="evil.example")[0] == 400
        assert fetch(LOOPBACK, server.port, f"{API}/status", host="localhost")[0] == 200


def test_the_host_rule_follows_the_settings_over_a_real_connection(
    start: Callable[..., Running],
) -> None:
    with start(allowed_hosts=["pi.example"]) as server:
        assert fetch(LOOPBACK, server.port, f"{API}/status", host="pi.example:8080")[0] == 200
        assert fetch(LOOPBACK, server.port, f"{API}/status", host="Pi.Example.")[0] == 200
        assert fetch(LOOPBACK, server.port, f"{API}/status", host="other.example")[0] == 400


@needs_websocket_library
def test_a_websocket_handshake_is_checked_over_a_real_connection(
    start: Callable[..., Running],
) -> None:
    with start() as server:
        port = server.port
        assert "101" in handshake(LOOPBACK, port)
        assert "101" in handshake(LOOPBACK, port, origin=f"http://localhost:{port}")
        assert "403" in handshake(LOOPBACK, port, origin="http://evil.example")
        assert "400" in handshake(LOOPBACK, port, host="evil.example")


# --- The stop --------------------------------------------------------------------------------


def test_the_stop_runs_the_lifespan_of_the_app(start: Callable[..., Running]) -> None:
    spies: list[SpyApp] = []

    def wrap(app: ASGIApp) -> ASGIApp:
        spies.append(SpyApp(app))
        return spies[0]

    with start(wrap=wrap):
        pass
    assert spies[0].lifespan == ["lifespan.startup.complete", "lifespan.shutdown.complete"]


def test_a_stop_before_the_server_starts_returns_at_once(app: FastAPI) -> None:
    runner = WebRunner(app, WebSettings(port=0))
    runner.request_stop("early")
    assert runner.run() == 0
    assert runner.stopped.is_set()


def test_a_failed_startup_gives_the_startup_failure_code() -> None:
    async def failing(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await receive()
            await send({"type": "lifespan.startup.failed", "message": "no start"})

    runner = WebRunner(failing, WebSettings(port=0))
    assert runner.run() == EXIT_STARTUP_FAILURE
    assert runner.stopped.is_set()
    assert not runner.started.is_set()


def test_a_taken_port_makes_run_raise_a_bind_error_that_names_the_address(app: FastAPI) -> None:
    (blocker,) = bind_sockets([LOOPBACK], 0)
    try:
        port = int(blocker.getsockname()[1])
        runner = WebRunner(app, WebSettings(port=port))
        with pytest.raises(BindError) as raised:
            runner.run()
        assert f"{LOOPBACK}:{port}" in str(raised.value)
    finally:
        blocker.close()


def test_the_url_needs_the_bound_port() -> None:
    runner = WebRunner(FastAPI(), WebSettings(port=0))
    with pytest.raises(RuntimeError, match="not bound"):
        _ = runner.url


def test_the_started_hook_runs_once_when_the_server_accepts_connections(
    start: Callable[..., Running],
) -> None:
    calls: list[float] = []
    with start(on_started=lambda: calls.append(time.monotonic())) as server:
        assert len(calls) == 1
        assert fetch(LOOPBACK, server.port, f"{API}/status")[0] == 200
        time.sleep(0.2)
        assert len(calls) == 1  # it never runs again


def test_a_failing_started_hook_is_logged_and_the_server_goes_on(
    start: Callable[..., Running], caplog: pytest.LogCaptureFixture
) -> None:
    def broken() -> None:
        raise RuntimeError("the hook broke")

    with caplog.at_level("ERROR"), start(on_started=broken) as server:
        assert fetch(LOOPBACK, server.port, f"{API}/status")[0] == 200
    assert any("on_started hook failed" in record.getMessage() for record in caplog.records)
    assert server.exit_code == 0


# --- Systemd ---------------------------------------------------------------------------------


class Messages:
    """The datagrams that a fake notify socket received, as lists of lines."""

    def __init__(self) -> None:
        self.items: list[list[str]] = []
        self._lock = threading.Lock()

    def __call__(self, data: bytes) -> None:
        with self._lock:
            self.items.append(data.decode("utf-8").split("\n"))

    def snapshot(self) -> list[list[str]]:
        with self._lock:
            return list(self.items)

    def count(self, line: str) -> int:
        return sum(1 for item in self.snapshot() if line in item)


def notifier_for(messages: Messages, interval_s: float = 0.2) -> SystemdNotifier:
    """A notifier that systemd would have configured: a watchdog at twice the interval."""
    usec = round(interval_s * 2 * 1e6)
    return SystemdNotifier(env={"WATCHDOG_USEC": str(usec)}, send=messages)


def test_systemd_hears_ready_then_heartbeats_then_stopping(start: Callable[..., Running]) -> None:
    messages = Messages()
    with start(notifier=notifier_for(messages)):
        assert wait_until(lambda: messages.count("WATCHDOG=1") >= 3, timeout_s=10)
    snapshot = messages.snapshot()
    assert snapshot[0][0] == "READY=1"
    assert snapshot[0][1].startswith("STATUS=serving 1 address")
    assert snapshot[-1] == ["STOPPING=1"]
    assert messages.count("READY=1") == 1
    assert messages.count("STOPPING=1") == 1
    kinds = [item[0] for item in snapshot]
    assert kinds.index("READY=1") < kinds.index("WATCHDOG=1") < kinds.index("STOPPING=1")


def test_ready_goes_out_when_the_server_accepts_connections(start: Callable[..., Running]) -> None:
    messages = Messages()
    with start(notifier=notifier_for(messages)) as server:
        # `started` is set when READY=1 goes out, and the server then answers at once.
        assert messages.count("READY=1") == 1
        assert fetch(LOOPBACK, server.port, f"{API}/status")[0] == 200


def test_without_a_watchdog_interval_systemd_hears_ready_and_stopping_only(
    start: Callable[..., Running],
) -> None:
    messages = Messages()
    with start(notifier=SystemdNotifier(env={}, send=messages)):
        time.sleep(0.3)
    assert [item[0] for item in messages.snapshot()] == ["READY=1", "STOPPING=1"]


class Wedge:
    """An ASGI app that stops answering the probe path while `wedged` is set."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.wedged = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/api/v1/profile":
            while self.wedged:
                await asyncio.sleep(0.01)
        await self.app(scope, receive, send)


def test_a_server_that_does_not_answer_its_own_probe_sends_no_heartbeat(
    start: Callable[..., Running],
) -> None:
    wedges: list[Wedge] = []

    def wrap(app: ASGIApp) -> ASGIApp:
        wedges.append(Wedge(app))
        return wedges[0]

    messages = Messages()
    notifier = notifier_for(messages, interval_s=0.1)
    with start(wrap=wrap, notifier=notifier, probe_timeout_s=0.2):
        (wedge,) = wedges
        assert wait_until(lambda: messages.count("WATCHDOG=1") >= 2, timeout_s=10)
        wedge.wedged = True
        time.sleep(0.5)  # a probe in flight finishes or times out
        before = messages.count("WATCHDOG=1")
        time.sleep(0.8)
        assert messages.count("WATCHDOG=1") == before  # no heartbeat while the server is wedged
        wedge.wedged = False
        assert wait_until(lambda: messages.count("WATCHDOG=1") > before, timeout_s=10)
    assert messages.snapshot()[-1] == ["STOPPING=1"]


def test_the_heartbeat_does_not_depend_on_core(
    start: Callable[..., Running], core: FakeCoreClient
) -> None:
    core.fail_with = CoreUnavailableError("down")
    messages = Messages()
    with start(notifier=notifier_for(messages)) as server:
        assert fetch(LOOPBACK, server.port, f"{API}/health")[0] == 503  # core is down
        assert wait_until(lambda: messages.count("WATCHDOG=1") >= 2, timeout_s=10)


if sys.platform != "win32":

    def notify_socket_roundtrip(start: Callable[..., Running]) -> list[str]:
        """Serve with a notifier that talks to a real datagram socket. Returns the first lines."""
        with tempfile.TemporaryDirectory(prefix="smon") as folder:
            path = str(Path(folder) / "notify")
            receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            receiver.bind(path)
            receiver.settimeout(10)
            received: list[str] = []

            def receive_one() -> None:
                received.append(receiver.recv(4096).decode("utf-8").split("\n")[0])

            try:
                env = {"NOTIFY_SOCKET": path, "WATCHDOG_USEC": "400000"}
                with start(notifier=SystemdNotifier(env=env)):
                    while received.count("WATCHDOG=1") < 2:
                        receive_one()
                while received[-1] != "STOPPING=1":
                    receive_one()
            finally:
                receiver.close()
        return received


def test_a_real_notify_socket_gets_the_datagrams(start: Callable[..., Running]) -> None:
    if sys.platform == "win32":
        pytest.skip("Windows has no datagram sockets for AF_UNIX")
    else:
        received = notify_socket_roundtrip(start)
        assert received[0] == "READY=1"
        assert received.count("WATCHDOG=1") >= 2
        assert received[-1] == "STOPPING=1"


# --- The probe -------------------------------------------------------------------------------


def serve_once(reply: bytes, *, answer: bool = True) -> tuple[int, threading.Thread]:
    """Listen on a free port, accept one connection, and send `reply` (or nothing)."""
    (listener,) = bind_sockets([LOOPBACK], 0)
    port = int(listener.getsockname()[1])

    def run() -> None:
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                connection.recv(4096)
                if answer:
                    connection.sendall(reply)
                else:
                    time.sleep(1.0)
        except OSError:
            pass
        finally:
            listener.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread


@pytest.mark.parametrize(
    "status", ["200 OK", "401 Unauthorized", "404 Not Found", "503 Service Unavailable"]
)
def test_the_probe_accepts_any_http_status_line(status: str) -> None:
    port, thread = serve_once(f"HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n".encode("ascii"))
    assert asyncio.run(probe_http(LOOPBACK, port, "/x", 2.0)) is True
    thread.join(5)


def test_the_probe_rejects_a_reply_that_is_not_http() -> None:
    port, thread = serve_once(b"SSH-2.0-nothing\r\n")
    assert asyncio.run(probe_http(LOOPBACK, port, "/x", 2.0)) is False
    thread.join(5)


def test_the_probe_fails_when_nobody_listens() -> None:
    assert asyncio.run(probe_http(LOOPBACK, free_port(), "/x", 0.5)) is False


def test_the_probe_fails_when_the_server_does_not_answer_in_time() -> None:
    port, thread = serve_once(b"", answer=False)
    started = time.monotonic()
    assert asyncio.run(probe_http(LOOPBACK, port, "/x", 0.2)) is False
    assert time.monotonic() - started < 0.9
    thread.join(5)
