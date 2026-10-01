"""Tell systemd that the process is alive: the `sd_notify` protocol, with the standard library.

A service that runs under systemd with `Type=notify` and `WatchdogSec=` reports to the manager
through a datagram socket. systemd names the socket in `NOTIFY_SOCKET`, and it gives the
watchdog interval in `WATCHDOG_USEC`. The service sends `READY=1` when it is ready, `WATCHDOG=1`
more often than the interval while it is healthy, and `STOPPING=1` when it shuts down. If the
heartbeats stop, systemd kills the process and starts it again.

`acquire`, `core`, and `web` share this module. `SystemdNotifier` does nothing unless
`NOTIFY_SOCKET` is set, so the same code runs under a test and on a development machine. A failure
to send is logged once and never raised: a service must not stop because the manager cannot hear it.

The protocol is in the `sd_notify(3)` manual page. A message is lines of `KEY=value`, joined by
newlines, in one datagram. A socket path that starts with `@` is in the abstract namespace.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from collections.abc import Callable, Mapping

_log = logging.getLogger(__name__)

if sys.platform != "win32":

    def _open_unix(address: str) -> socket.socket:
        target = "\0" + address[1:] if address.startswith("@") else address
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC)
        try:
            sock.connect(target)
        except OSError:
            sock.close()
            raise
        return sock

else:

    def _open_unix(address: str) -> socket.socket:
        raise OSError("systemd notification needs Unix sockets")


class SystemdNotifier:
    """Send state to systemd. Pass `env` and `send` to test it without a socket."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        send: Callable[[bytes], object] | None = None,
        pid: int | None = None,
    ) -> None:
        environment = os.environ if env is None else env
        self._address = environment.get("NOTIFY_SOCKET", "")
        self._send: Callable[[bytes], object] | None = send
        self._socket: socket.socket | None = None
        self._failed = False
        self._last_status: str | None = None
        self.watchdog_interval_s: float | None = None
        usec = environment.get("WATCHDOG_USEC", "")
        owner = environment.get("WATCHDOG_PID", "")
        mine = os.getpid() if pid is None else pid
        if usec.isdigit() and int(usec) > 0 and (not owner or owner == str(mine)):
            self.watchdog_interval_s = int(usec) / 1e6 / 2  # systemd advises half the timeout

    @property
    def enabled(self) -> bool:
        """Whether the notifier sends anything: systemd named a socket, or a sender was given."""
        return bool(self._address) or self._send is not None

    def _deliver(self, lines: list[str]) -> None:
        if not self.enabled or self._failed:
            return
        try:
            if self._send is None:
                self._socket = _open_unix(self._address)
                self._send = self._socket.send
            self._send("\n".join(lines).encode("utf-8"))
        except OSError as error:
            self._failed = True
            _log.warning("cannot notify systemd, so notifications stop: %s", error)

    def ready(self, status: str | None = None) -> None:
        """Say that startup is done."""
        if status:
            self._last_status = status
        self._deliver(["READY=1", *([f"STATUS={status}"] if status else [])])

    def watchdog(self) -> None:
        """Say that the service is healthy. Call it more often than the watchdog interval."""
        self._deliver(["WATCHDOG=1"])

    def status(self, text: str) -> None:
        """Set the one-line status that `systemctl status` shows."""
        self._last_status = text
        self._deliver([f"STATUS={text.replace(chr(10), ' ')}"])

    def status_changed(self, text: str) -> bool:
        """Set the status when it differs from the last one. Returns whether it sent anything.

        Call it as often as you like with the current state: a message goes out only when the text
        changes.
        """
        if text == self._last_status:
            return False
        self.status(text)
        return True

    def stopping(self) -> None:
        """Say that the service is shutting down."""
        self._deliver(["STOPPING=1"])

    def close(self) -> None:
        """Close the socket. Later messages are not sent."""
        sock, self._socket = self._socket, None
        self._failed = True
        if sock is not None:
            sock.close()
