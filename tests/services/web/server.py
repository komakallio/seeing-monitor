"""Helpers for the tests that run a real server: a thread, plain HTTP requests, and a handshake.

The tests of the runner, the commands, and the demo start `WebRunner` on loopback addresses and
talk to it over real sockets. `http.client` and raw sockets keep the requests exact: the `Host` and
`Origin` headers are whatever the test says.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import socket
import threading
import time
from typing import Any

import pytest
from starlette.types import ASGIApp

from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.runner import BindError, WebRunner, bind_sockets, format_address

LOOPBACK = "127.0.0.1"
OTHER_LOOPBACK = "127.0.0.2"
IPV6_LOOPBACK = "::1"
API = "/api/v1"


def can_bind(address: str) -> bool:
    try:
        for sock in bind_sockets([address], 0):
            sock.close()
    except BindError:
        return False
    return True


needs_second_loopback = pytest.mark.skipif(
    not can_bind(OTHER_LOOPBACK), reason="this system cannot bind a second loopback address"
)
needs_ipv6 = pytest.mark.skipif(
    not can_bind(IPV6_LOOPBACK), reason="this system has no IPv6 loopback address"
)
needs_websocket_library = pytest.mark.skipif(
    not any(importlib.util.find_spec(name) is not None for name in ("websockets", "wsproto")),
    reason="uvicorn needs the websockets or wsproto package to accept a WebSocket",
)


def assert_nobody_listens(address: str, port: int) -> None:
    """Check that a connection fails. Windows retries a refused loopback connection for 2 s, so a
    refusal can show as a timeout there. An open socket would accept, even with no one to serve it.
    """
    try:
        socket.create_connection((address, port), timeout=3).close()
    except (ConnectionRefusedError, TimeoutError):
        return
    pytest.fail(f"something still listens on {format_address(address, port)}")


def free_port() -> int:
    sockets = bind_sockets([LOOPBACK], 0)
    try:
        return int(sockets[0].getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()


class Running:
    """Run a `WebRunner` in a thread. Leaving the block stops it and checks that it stopped."""

    def __init__(self, app: ASGIApp, settings: WebSettings, **options: Any) -> None:
        self.runner = WebRunner(app, settings, **options)
        self.exit_code: int | None = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="test-web-runner", daemon=True)

    def _run(self) -> None:
        try:
            self.exit_code = self.runner.run()
        except BaseException as error:  # the test reports it
            self.error = error

    @property
    def port(self) -> int:
        assert self.runner.port is not None
        return self.runner.port

    def __enter__(self) -> Running:
        self._thread.start()
        wait_started(self.runner, self._thread, lambda: self.error)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.runner.request_stop("test")
        self._thread.join(20)
        assert not self._thread.is_alive(), "the runner did not stop"


def wait_started(runner: WebRunner, thread: threading.Thread, error: Any) -> None:
    """Wait until the runner accepts connections. Fails when its thread ends first."""
    deadline = time.monotonic() + 20
    while not runner.started.is_set():
        assert thread.is_alive(), f"the runner ended at start: {error()!r}"
        assert time.monotonic() < deadline, "the runner did not start"
        time.sleep(0.01)


def fetch(
    address: str,
    port: int,
    path: str,
    *,
    host: str | None = None,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """One HTTP request over a real connection. The `Host` is the address unless given."""
    import http.client

    connection = http.client.HTTPConnection(address, port, timeout=10)
    try:
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        connection.putheader("Host", host if host is not None else format_address(address, port))
        for name, value in (headers or {}).items():
            connection.putheader(name, value)
        if body is not None:
            connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        reply_headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, reply_headers, response.read()
    finally:
        connection.close()


def handshake(
    address: str, port: int, *, host: str | None = None, origin: str | None = None
) -> str:
    """Send a WebSocket upgrade over a real connection and return the status line of the answer."""
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    lines = [
        f"GET {API}/alignment/stream HTTP/1.1",
        f"Host: {host if host is not None else format_address(address, port)}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    if origin is not None:
        lines.append(f"Origin: {origin}")
    request = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
    with socket.create_connection((address, port), timeout=10) as sock:
        sock.sendall(request)
        reply = b""
        while b"\r\n" not in reply:
            chunk = sock.recv(1024)
            if not chunk:
                break
            reply += chunk
    return reply.split(b"\r\n", 1)[0].decode("latin-1")
