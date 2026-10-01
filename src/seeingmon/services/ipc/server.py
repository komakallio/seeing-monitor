"""The listening side of the connection layer.

An `IpcServer` listens at one `Endpoint` and serves named channels. A client connects, proves
that it holds the connection key (see `seeingmon.services.ipc.handshake`), and names the
channel it wants in a `hello` message. The server hands the connection to the handler of that
channel: an `RpcService` for requests and answers, or a `StreamService` for a frame stream.
One server and one address carry every channel, and each channel gets its own connection, so
the messages of different channels share nothing.

The accept loop never waits on a client. It accepts a connection and passes it to a short-lived
thread that runs the handshake, and it refuses new connections while `max_pending` handshakes
run, so a client that stays silent cannot lock out the others.

**Unix sockets.** The server removes a stale socket file that a killed process left behind. It
refuses to start when another process still listens at the path, and it refuses to remove a
file that is not a socket. The socket file is private to the user (mode 0600), and so is the
directory that the server creates for it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import stat
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Protocol

from seeingmon.services.ipc.codec import (
    as_mapping,
    decode_json,
    encode_exception,
    encode_json,
    get_int,
    get_str,
)
from seeingmon.services.ipc.endpoint import FAMILY_UNIX, Endpoint
from seeingmon.services.ipc.errors import (
    IpcAddressInUseError,
    IpcAuthError,
    IpcClosedError,
    IpcError,
    IpcProtocolError,
)
from seeingmon.services.ipc.handshake import serve_handshake
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.wire import Wire

HELLO_VERSION = 1
MAX_HELLO_BYTES = 64 * 1024
DEFAULT_MESSAGE_BYTES = 1024 * 1024

_log = logging.getLogger(__name__)
_UMASK_LOCK = threading.Lock()


def _noop() -> None:
    return None


@dataclass(slots=True)
class Accepted:
    """What a channel handler returns when it takes a connection.

    `reply` goes to the client in the answer to its `hello`. The server sends the answer, and
    then calls `activate`, which is the place to start threads and to register the connection:
    the client cannot send or receive channel messages before it has the answer. If the server
    cannot send the answer, it calls `abort` instead, and the handler undoes what `accept` did.
    """

    reply: Mapping[str, Any] = field(default_factory=dict)
    activate: Callable[[], None] = _noop
    abort: Callable[[], None] = _noop


class ChannelHandler(Protocol):
    """The server side of one channel."""

    def accept(self, wire: Wire, params: Mapping[str, Any]) -> Accepted:
        """Take a connection that asked for this channel with `params`.

        Raise an `IpcError` to refuse it. The message of the error goes to the client. The
        handler owns the wire from here, and it closes the wire when the channel ends.
        """
        ...


@contextlib.contextmanager
def _private_umask() -> Iterator[None]:
    """Create files with mode 0600 while the block runs (a no-op on Windows)."""
    if os.name != "posix":
        yield
        return
    with _UMASK_LOCK:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)


def _claim_unix_path(path: str) -> None:
    """Make the socket path free: create its directory, and remove a stale socket file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(info.st_mode):
        raise IpcAddressInUseError("the socket path exists and is not a socket")
    try:
        probe = Client(path, FAMILY_UNIX)
    except (ConnectionRefusedError, FileNotFoundError):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)  # nobody listens: the file is left over from a process that died
        return
    except OSError as error:
        raise IpcAddressInUseError(f"cannot tell whether the socket is in use: {error}") from None
    probe.close()
    raise IpcAddressInUseError("another process already listens at the socket path")


@dataclass(slots=True)
class ServerStats:
    """Counters of an `IpcServer`."""

    accepted: int = 0
    activated: int = 0
    auth_failures: int = 0
    hello_failures: int = 0
    refused_busy: int = 0


class IpcServer:
    """Listen at an endpoint and hand authenticated connections to channel handlers."""

    def __init__(
        self,
        endpoint: Endpoint,
        key: ConnectionKey,
        channels: Mapping[str, ChannelHandler],
        *,
        handshake_timeout_s: float = 5.0,
        max_pending: int = 8,
        backlog: int = 8,
        name: str = "ipc",
    ) -> None:
        self._configured = endpoint
        self._endpoint = endpoint
        self._key = key
        self._channels = dict(channels)
        self._handshake_timeout_s = handshake_timeout_s
        self._max_pending = max_pending
        self._backlog = backlog
        self._name = name
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listener: Listener | None = None
        self._thread: threading.Thread | None = None
        self._pending: set[Wire] = set()
        self.stats = ServerStats()

    @property
    def endpoint(self) -> Endpoint:
        """The address that the server listens at. For a TCP endpoint with port 0, the real port."""
        return self._endpoint

    @property
    def running(self) -> bool:
        """Whether the accept loop runs."""
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self) -> Endpoint:
        """Bind the address and start accepting. Raises `IpcAddressInUseError` if it is taken."""
        if self._listener is not None:
            raise RuntimeError("the server already started")
        endpoint = self._configured
        if endpoint.path is not None:
            _claim_unix_path(endpoint.path)
        try:
            with _private_umask():
                listener = Listener(endpoint.address, family=endpoint.family, backlog=self._backlog)
        except OSError as error:
            raise IpcAddressInUseError(
                f"cannot listen at {endpoint}: {error.strerror or type(error).__name__}"
            ) from None
        if endpoint.path is not None:
            with contextlib.suppress(OSError):
                os.chmod(endpoint.path, 0o600)
        self._listener = listener
        bound = listener.address
        if isinstance(endpoint.address, tuple) and isinstance(bound, tuple):
            # TCP with a free port: report the port that the system picked.
            self._endpoint = Endpoint((str(bound[0]), int(bound[1])), endpoint.family)
        self._thread = threading.Thread(
            target=self._accept_loop, args=(listener,), name=f"{self._name}-accept", daemon=True
        )
        self._thread.start()
        return self._endpoint

    def stop(self) -> None:
        """Stop accepting, close the connections that have not finished their handshake."""
        self._stop.set()
        listener, self._listener = self._listener, None
        thread, self._thread = self._thread, None
        if listener is None:
            return
        self._wake(self._endpoint)  # `accept` cannot be interrupted, so give it a connection
        if thread is not None and thread is not threading.current_thread():
            thread.join(2.0)
        with contextlib.suppress(Exception):
            listener.close()
        with self._lock:
            pending = list(self._pending)
        for wire in pending:
            wire.close("the server stopped")

    @staticmethod
    def _wake(endpoint: Endpoint) -> None:
        with contextlib.suppress(OSError, EOFError):  # the listener may be closed already
            Client(endpoint.address, family=endpoint.family).close()

    def _accept_loop(self, listener: Listener) -> None:
        while not self._stop.is_set():
            try:
                connection = listener.accept()
            except (OSError, EOFError, ValueError):
                if self._stop.is_set():
                    return
                _log.exception("%s: accept failed", self._name)
                self._stop.wait(0.1)
                continue
            if self._stop.is_set():
                connection.close()
                return
            wire = Wire(connection, max_message_bytes=DEFAULT_MESSAGE_BYTES, name=self._name)
            with self._lock:
                busy = len(self._pending) >= self._max_pending
                if not busy:
                    self._pending.add(wire)
                    self.stats.accepted += 1
                else:
                    self.stats.refused_busy += 1
            if busy:
                wire.close("too many connections are waiting for their handshake")
                continue
            threading.Thread(
                target=self._serve, args=(wire,), name=f"{self._name}-handshake", daemon=True
            ).start()

    def _serve(self, wire: Wire) -> None:
        handed_over = False
        try:
            serve_handshake(wire, self._key, timeout_s=self._handshake_timeout_s)
            handed_over = self._admit(wire)
        except IpcAuthError as error:
            self.stats.auth_failures += 1
            _log.warning("%s: refused a connection: %s", self._name, error)
        except IpcError as error:
            self.stats.hello_failures += 1
            _log.warning("%s: dropped a connection: %s", self._name, error)
        except Exception:
            _log.exception("%s: a connection failed", self._name)
        finally:
            with self._lock:
                self._pending.discard(wire)
            if not handed_over:
                wire.close()

    def _read_hello(self, wire: Wire) -> tuple[str, Mapping[str, Any]]:
        raw = wire.recv(self._handshake_timeout_s, max_bytes=MAX_HELLO_BYTES)
        if raw is None:
            raise IpcProtocolError("the client sent no hello in time")
        data = as_mapping(decode_json(raw), "hello")
        if get_int(data, "v", "hello") != HELLO_VERSION:
            raise IpcProtocolError("the client speaks another protocol version")
        params = as_mapping(data.get("params", {}), "hello.params")
        return get_str(data, "channel", "hello"), params

    def _refuse(self, wire: Wire, error: Exception) -> None:
        with contextlib.suppress(IpcClosedError):
            wire.send(encode_json({"ok": False, "error": encode_exception(error)}))

    def _admit(self, wire: Wire) -> bool:
        """Read the hello, and give the wire to its channel. Returns whether the channel took it."""
        try:
            channel, params = self._read_hello(wire)
            handler = self._channels.get(channel)
            if handler is None:
                raise IpcProtocolError(f"there is no channel named {channel!r}")
            accepted = handler.accept(wire, params)
        except IpcError as error:
            self.stats.hello_failures += 1
            self._refuse(wire, error)
            _log.warning("%s: refused a hello: %s", self._name, error)
            return False
        try:
            wire.send(encode_json({"ok": True, "v": HELLO_VERSION, "reply": dict(accepted.reply)}))
        except IpcClosedError:
            accepted.abort()
            return False
        self.stats.activated += 1
        accepted.activate()
        return True
