"""Serve the app: one listening socket for each address, one event loop, and systemd.

`WebRunner` binds the sockets, hands them to uvicorn, and keeps the process honest toward systemd:

- **Sockets.** The runner opens one listening socket for each distinct address of `bind_address`
  and `extra_bind_addresses`, all on the same port, and uvicorn serves them from one event loop.
  A wildcard address (`0.0.0.0` or `::`) listens on every interface of its family, whatever
  address the device gets later. An IPv6 socket accepts IPv6 only (`IPV6_V6ONLY`), so the IPv4
  and IPv6 addresses of one interface stay two separate listeners that the configuration names. On
  Windows a socket binds exclusively, so no other process can share the port. If an address
  cannot be bound, the runner closes the sockets that it opened, and `BindError` names the
  address. Port 0 asks the system for a free port, which every socket then shares.
- **Systemd.** The runner sends `READY=1` when the server accepts connections, `WATCHDOG=1` at half
  the watchdog interval, and `STOPPING=1` when the shutdown begins. A heartbeat goes out only if a
  real HTTP request to the first listening address (a read of the profile, which needs the event
  loop and a worker thread) gets an answer, so a wedged server stops sending and systemd restarts
  it. For a wildcard the request goes to the loopback address of its family. The probe never
  touches the store or `core`: an outage of either must not restart `web`.
- **Shutdown.** A stop request or a signal makes uvicorn stop accepting connections, close the live
  WebSocket connections, and wait up to `shutdown_timeout_s` for the rest. The lifespan of the app
  then closes the alignment hub. `run` returns 0 after a clean stop, and `EXIT_STARTUP_FAILURE`
  when the server did not start.

The runner sets no signal handler of its own. In the main thread, uvicorn catches SIGINT and
SIGTERM while it serves. A handler that the caller installed before `run` stays in place and
receives the signal again after the shutdown, so the caller can clean up and exit with a code.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import socket
import sys
import threading
import time
from collections.abc import Callable, Sequence

import uvicorn
from starlette.types import ASGIApp

from seeingmon.services.acquire.notify import SystemdNotifier
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.netaddr import WILDCARD_V4, WILDCARD_V6, connect_address

_log = logging.getLogger(__name__)

EXIT_STARTUP_FAILURE = 3
BACKLOG = 128
PROBE_PATH = "/api/v1/profile"
PROBE_TIMEOUT_S = 10.0
POLL_S = 0.05
WS_MAX_MESSAGE_BYTES = 64 * 1024
_WILDCARD_SCOPE = {WILDCARD_V4: "every IPv4 interface", WILDCARD_V6: "every IPv6 interface"}


class BindError(OSError):
    """The process cannot listen on an address. The message names the address."""


def format_address(address: str, port: int) -> str:
    """The address and port as text: `192.0.2.5:8080` or `[2001:db8::5]:8080`."""
    return f"[{address}]:{port}" if ":" in address else f"{address}:{port}"


def url_for(address: str, port: int) -> str:
    """The URL of the UI on one listening address."""
    return f"http://{format_address(address, port)}/"


def _listening_socket(address: str, port: int, backlog: int) -> socket.socket:
    shown = format_address(address, port)
    try:
        family, kind, proto, _name, sockaddr = socket.getaddrinfo(
            address,
            port,
            type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE | socket.AI_NUMERICHOST,
        )[0]
        sock = socket.socket(family, kind, proto)
    except OSError as error:
        raise BindError(f"cannot listen on {shown}: {error.strerror or 'bad address'}") from None
    try:
        if sys.platform == "win32":
            # SO_REUSEADDR on Windows lets another process bind the same port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind(sockaddr)
        sock.listen(backlog)
    except OSError as error:
        sock.close()
        raise BindError(
            f"cannot listen on {shown}: {error.strerror or type(error).__name__}"
        ) from None
    return sock


def bind_sockets(
    addresses: Sequence[str], port: int, *, backlog: int = BACKLOG
) -> list[socket.socket]:
    """Open one listening socket for each address, all on the same port.

    Port 0 asks the system for a free port: the first socket takes one, and the others use it too.
    Raises `BindError` that names the first address that fails. The function closes the sockets
    that it had opened before it raises.
    """
    sockets: list[socket.socket] = []
    chosen = port
    try:
        for address in addresses:
            sock = _listening_socket(address, chosen, backlog)
            sockets.append(sock)
            if chosen == 0:
                chosen = int(sock.getsockname()[1])
    except BindError:
        for sock in sockets:
            sock.close()
        raise
    return sockets


async def probe_http(address: str, port: int, path: str, timeout_s: float) -> bool:
    """Whether a server answers an HTTP request on this address. Any status line counts.

    The request goes through a real connection, so it proves that the event loop and the HTTP
    stack work. The `Host` is the address itself, which the host check always allows.
    """
    host = f"[{address}]" if ":" in address else address
    request = (
        f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n"
        "User-Agent: seeingmon-watchdog\r\nAccept: application/json\r\n\r\n"
    ).encode("ascii")
    try:
        async with asyncio.timeout(timeout_s):
            reader, writer = await asyncio.open_connection(address, port)
            try:
                writer.write(request)
                await writer.drain()
                line = await reader.readline()
            finally:
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()
    except (OSError, TimeoutError):
        return False
    return line.startswith(b"HTTP/1.")


class WebRunner:
    """Serve an ASGI app on the addresses of the settings. See the module documentation."""

    def __init__(
        self,
        app: ASGIApp,
        settings: WebSettings,
        *,
        port: int | None = None,
        notifier: SystemdNotifier | None = None,
        probe_path: str = PROBE_PATH,
        probe_timeout_s: float = PROBE_TIMEOUT_S,
        watchdog_interval_s: float | None = None,
        on_started: Callable[[], None] | None = None,
    ) -> None:
        self._app = app
        self._settings = settings
        self._addresses = settings.listen_addresses()
        self._requested_port = settings.port if port is None else port
        self._notifier = notifier if notifier is not None else SystemdNotifier()
        self._probe_path = probe_path
        self._probe_timeout_s = probe_timeout_s
        self._watchdog_interval_s = watchdog_interval_s or self._notifier.watchdog_interval_s
        self._on_started = on_started
        self._server: uvicorn.Server | None = None
        self._stop_requested = False
        self.port: int | None = None
        self.started = threading.Event()
        self.stopped = threading.Event()

    @property
    def addresses(self) -> tuple[str, ...]:
        """The addresses that the runner listens on, in the order of the settings."""
        return self._addresses

    @property
    def url(self) -> str:
        """The URL of the UI on the first address, as a program on this device reaches it (a
        wildcard gives the loopback address). It needs the port, so call it after `run` has bound
        the sockets (see `started`)."""
        if self.port is None:
            raise RuntimeError("the runner has not bound its sockets yet")
        return url_for(connect_address(self._addresses[0]), self.port)

    def request_stop(self, reason: str = "requested") -> None:
        """Ask the server to shut down. Safe to call from another thread or a signal handler."""
        self._stop_requested = True
        server = self._server
        if server is not None:
            _log.info("stopping the web server: %s", reason)
            server.should_exit = True

    def run(self) -> int:
        """Bind the sockets, serve until a stop, and return the exit code.

        Raises `BindError` when an address cannot be bound.
        """
        sockets = bind_sockets(self._addresses, self._requested_port)
        self.port = int(sockets[0].getsockname()[1])
        try:
            for address in self._addresses:
                scope = _WILDCARD_SCOPE.get(address)
                shown = format_address(address, self.port)
                _log.info("listening on %s%s", shown, f" ({scope})" if scope else "")
            try:
                return asyncio.run(self._main(sockets))
            except SystemExit as stop:  # uvicorn exits with a code when the startup fails
                return stop.code if isinstance(stop.code, int) else EXIT_STARTUP_FAILURE
        finally:
            for sock in sockets:
                sock.close()
            self._notifier.close()
            self.stopped.set()

    async def _main(self, sockets: list[socket.socket]) -> int:
        assert self.port is not None
        settings = self._settings
        config = uvicorn.Config(
            self._app,
            host=self._addresses[0],
            port=self.port,
            log_config=None,
            access_log=settings.access_log,
            lifespan="on",
            proxy_headers=False,
            server_header=False,
            ws_max_size=WS_MAX_MESSAGE_BYTES,
            timeout_graceful_shutdown=max(1, math.ceil(settings.shutdown_timeout_s)),
        )
        server = uvicorn.Server(config)
        self._server = server
        if self._stop_requested:
            return 0
        serving = asyncio.ensure_future(server.serve(sockets=sockets))
        watching = asyncio.ensure_future(self._watch(server))
        try:
            await serving
        finally:
            watching.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watching
        return 0 if server.started else EXIT_STARTUP_FAILURE

    def _announce(self) -> None:
        """Call the `on_started` hook. A failure in the hook never stops the server."""
        if self._on_started is None:
            return
        try:
            self._on_started()
        except Exception:
            _log.exception("the on_started hook failed")

    async def _watch(self, server: uvicorn.Server) -> None:
        """Tell systemd about the start, the heartbeat, and the stop, and set `started`."""
        assert self.port is not None
        notifier = self._notifier
        interval = self._watchdog_interval_s
        started = False
        stopping = False
        next_beat = 0.0
        beating = True
        while True:
            if not started and server.started:
                started = True
                self.started.set()
                notifier.ready(f"serving {len(self._addresses)} address(es)")
                self._announce()
                next_beat = time.monotonic()
            if server.should_exit and not stopping:
                stopping = True
                notifier.stopping()
            if started and not stopping and interval is not None and time.monotonic() >= next_beat:
                alive = await probe_http(
                    connect_address(self._addresses[0]),
                    self.port,
                    self._probe_path,
                    self._probe_timeout_s,
                )
                if alive:
                    notifier.watchdog()
                    beating = True
                elif beating:
                    beating = False
                    _log.warning("the web server does not answer its own probe: no heartbeat")
                next_beat += interval
                next_beat = max(next_beat, time.monotonic() - interval)
            await asyncio.sleep(POLL_S)
