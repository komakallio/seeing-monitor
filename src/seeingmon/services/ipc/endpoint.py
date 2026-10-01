"""Where a service listens: a Unix socket path on Linux, a named pipe on Windows.

The connection layer uses `multiprocessing.connection` with `AF_UNIX` on Linux and `AF_PIPE`
on Windows. An `Endpoint` pairs an address with its family and checks both, so a mistake
shows up when the service starts and not when the first client connects.

Configuration gives an address as text. An empty value selects the default for the platform:

- Linux: `<runtime dir>/<role>.sock`. The runtime directory is the first entry of
  `RUNTIME_DIRECTORY` (which systemd sets for `RuntimeDirectory=`), else
  `$XDG_RUNTIME_DIR/seeingmon`, else `/run/seeingmon`.
- Windows: the pipe `\\\\.\\pipe\\seeingmon-<role>`.

On Windows, a value that does not start with a backslash is a pipe name, and the layer adds
the `\\\\.\\pipe\\` prefix. Only local pipes are allowed.

`Endpoint.loopback` builds a TCP endpoint on the loopback interface. It exists for tests,
which use it to exercise the socket code path on every platform. Configuration cannot select
it.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from seeingmon.services.ipc.errors import IpcConfigError

FAMILY_UNIX = "AF_UNIX"
FAMILY_PIPE = "AF_PIPE"
FAMILY_INET = "AF_INET"

PIPE_PREFIX = "\\\\.\\pipe\\"
PIPE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MAX_UNIX_PATH_BYTES = 107  # sun_path holds 108 bytes on Linux, and the last one is the NUL
DEFAULT_UNIX_DIR = "/run/seeingmon"
LOOPBACK_HOST = "127.0.0.1"


@dataclass(frozen=True, slots=True)
class Endpoint:
    """An address and its `multiprocessing.connection` family."""

    address: str | tuple[str, int]
    family: str

    def __post_init__(self) -> None:
        if self.family == FAMILY_INET:
            pair: Any = self.address  # a caller may pass anything, whatever the annotation says
            if not (
                isinstance(pair, tuple)
                and len(pair) == 2
                and isinstance(pair[0], str)
                and isinstance(pair[1], int)
            ):
                raise IpcConfigError("a TCP endpoint is a (host, port) pair")
            return
        if not isinstance(self.address, str) or not self.address:
            raise IpcConfigError("an address is a non-empty string")
        if "\0" in self.address:
            raise IpcConfigError("an address must not contain a NUL character")
        if self.family == FAMILY_UNIX:
            if len(self.address.encode("utf-8")) > MAX_UNIX_PATH_BYTES:
                raise IpcConfigError(
                    f"a socket path is at most {MAX_UNIX_PATH_BYTES} bytes long; "
                    "choose a shorter path"
                )
        elif self.family == FAMILY_PIPE:
            if not self.address.startswith(PIPE_PREFIX) or not PIPE_NAME_PATTERN.fullmatch(
                self.address[len(PIPE_PREFIX) :]
            ):
                raise IpcConfigError(
                    f"a pipe address is {PIPE_PREFIX}<name>, where the name has 1 to 128 "
                    "letters, digits, '.', '_', or '-'"
                )
        else:
            raise IpcConfigError(f"unsupported address family {self.family!r}")

    def __str__(self) -> str:
        if isinstance(self.address, tuple):
            return f"{self.address[0]}:{self.address[1]}"
        return self.address

    @property
    def path(self) -> str | None:
        """The socket file of a Unix endpoint, and `None` for the other families."""
        if self.family == FAMILY_UNIX and isinstance(self.address, str):
            return self.address
        return None

    @classmethod
    def loopback(cls, port: int = 0) -> Endpoint:
        """A TCP endpoint on the loopback interface. Port 0 lets the system pick one."""
        return cls((LOOPBACK_HOST, port), FAMILY_INET)

    @classmethod
    def parse(cls, text: str, *, platform: str | None = None) -> Endpoint:
        """Turn a configured address into an endpoint for the platform.

        `platform` is a `sys.platform` value, and the default is the running platform.
        """
        platform = sys.platform if platform is None else platform
        value = text.strip()
        if not value:
            raise IpcConfigError("the address is empty")
        if platform == "win32":
            if value.startswith("\\\\"):
                return cls(value, FAMILY_PIPE)
            return cls(PIPE_PREFIX + value, FAMILY_PIPE)
        if value.startswith("\\\\"):
            raise IpcConfigError("a named-pipe address works only on Windows")
        return cls(value, FAMILY_UNIX)

    @classmethod
    def default(
        cls, role: str, *, platform: str | None = None, env: Mapping[str, str] | None = None
    ) -> Endpoint:
        """The default endpoint of a service. `role` is `acquire` or `core`."""
        if not PIPE_NAME_PATTERN.fullmatch(role):
            raise IpcConfigError(f"not a valid role name: {role!r}")
        platform = sys.platform if platform is None else platform
        if platform == "win32":
            return cls(f"{PIPE_PREFIX}seeingmon-{role}", FAMILY_PIPE)
        environment = os.environ if env is None else env
        runtime = environment.get("RUNTIME_DIRECTORY", "").split(":")[0]
        if not runtime:
            xdg = environment.get("XDG_RUNTIME_DIR", "")
            runtime = f"{xdg}/seeingmon" if xdg else DEFAULT_UNIX_DIR
        return cls(f"{runtime.rstrip('/')}/{role}.sock", FAMILY_UNIX)

    @classmethod
    def from_setting(
        cls,
        text: str,
        role: str,
        *,
        platform: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Endpoint:
        """The configured address, or the default of the role when the setting is empty."""
        if text.strip():
            return cls.parse(text, platform=platform)
        return cls.default(role, platform=platform, env=env)
