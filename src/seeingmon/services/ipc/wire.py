"""One connection as a pipe of byte messages.

A `Wire` wraps a `multiprocessing.connection` connection and uses only `send_bytes`,
`recv_bytes`, and `poll`. It never calls `send` or `recv`, which pickle, so a peer cannot make
this process load an object.

**Threads.** Any thread may send: a lock keeps messages whole. One thread at a time should
receive. A receive waits in slices of 0.1 s, so `close` from another thread takes effect within
a slice, on every platform. (Closing a socket does not wake a blocked read on Linux.)

**Limits.** Every receive has a maximum message size, and an oversized message closes the
wire. A connection that has not authenticated yet gets a few hundred bytes.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.services.ipc.errors import IpcClosedError, IpcProtocolError

POLL_SLICE_S = 0.1
CLOSE_WAIT_S = 1.0

_log = logging.getLogger(__name__)
_REAL_CLOCK = SystemClock()


class RawConnection(Protocol):
    """The part of `multiprocessing.connection.Connection` that a `Wire` uses."""

    def send_bytes(self, buf: bytes | bytearray | memoryview) -> None: ...

    def recv_bytes(self, maxlength: int | None = None) -> bytes: ...

    def poll(self, timeout: float | None = 0.0) -> bool: ...

    def close(self) -> None: ...


class Wire:
    """A connection that sends and receives byte messages and nothing else."""

    def __init__(
        self,
        connection: RawConnection,
        *,
        max_message_bytes: int,
        name: str = "wire",
        clock: Clock | None = None,
    ) -> None:
        self._conn = connection
        self._max_message_bytes = max_message_bytes
        self.name = name
        self._clock = _REAL_CLOCK if clock is None else clock
        self._send_lock = threading.RLock()
        self._recv_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._closing = False
        self._reason = "the connection is closed"
        self.bytes_sent = 0
        self.bytes_received = 0
        self.messages_sent = 0
        self.messages_received = 0

    @property
    def closed(self) -> bool:
        """Whether the wire is closed, or closing."""
        return self._closing

    @property
    def reason(self) -> str:
        """Why the wire closed, in words."""
        return self._reason

    @property
    def max_message_bytes(self) -> int:
        """The largest message that `recv` accepts without its own `max_bytes`."""
        return self._max_message_bytes

    @max_message_bytes.setter
    def max_message_bytes(self, value: int) -> None:
        self._max_message_bytes = value

    def send(self, data: bytes | bytearray | memoryview) -> None:
        """Send one message. Raises `IpcClosedError` when the connection is gone."""
        if len(data) > self._max_message_bytes:
            raise IpcProtocolError(f"a message of {len(data)} bytes exceeds the limit")
        with self._send_lock:
            if self._closing:
                raise IpcClosedError(self._reason)
            try:
                self._conn.send_bytes(data)
            except (OSError, ValueError, EOFError) as error:
                self._fail(f"the peer went away while sending ({type(error).__name__})")
                raise IpcClosedError(self._reason) from None
            self.bytes_sent += len(data)
            self.messages_sent += 1

    def recv(self, timeout_s: float | None, *, max_bytes: int | None = None) -> bytes | None:
        """Receive one message, or return `None` when `timeout_s` passes without one.

        `timeout_s=None` waits until a message arrives or the wire closes. Raises
        `IpcClosedError` when the peer closes the connection, and `IpcProtocolError` when a
        message exceeds `max_bytes` (default: the limit of the wire). The wire closes in both
        cases.
        """
        limit = self._max_message_bytes if max_bytes is None else max_bytes
        started_ns = self._clock.monotonic_ns()
        with self._recv_lock:
            while True:
                if self._closing:
                    raise IpcClosedError(self._reason)
                if timeout_s is None:
                    slice_s = POLL_SLICE_S
                else:
                    elapsed_s = (self._clock.monotonic_ns() - started_ns) / NS_PER_S
                    slice_s = min(POLL_SLICE_S, max(0.0, timeout_s - elapsed_s))
                try:
                    if self._conn.poll(slice_s):
                        data = self._conn.recv_bytes(limit)
                        self.bytes_received += len(data)
                        self.messages_received += 1
                        return data
                except EOFError:
                    self._fail("the peer closed the connection")
                    raise IpcClosedError(self._reason) from None
                except (OSError, ValueError) as error:
                    if "bad message length" in str(error):
                        self._fail("the peer sent a message over the size limit")
                        raise IpcProtocolError(self._reason) from None
                    self._fail(f"the connection broke ({type(error).__name__})")
                    raise IpcClosedError(self._reason) from None
                if timeout_s is not None:
                    elapsed_s = (self._clock.monotonic_ns() - started_ns) / NS_PER_S
                    if elapsed_s >= timeout_s:
                        return None

    def _fail(self, reason: str) -> None:
        with self._state_lock:
            if self._closing:
                return
            self._closing = True
            self._reason = reason
        self._close_connection()

    def close(self, reason: str = "the connection is closed") -> None:
        """Close the connection. A thread that waits in `recv` notices within 0.1 s."""
        with self._state_lock:
            if self._closing:
                return
            self._closing = True
            self._reason = reason
        # Let a thread that is inside `recv` or `send` leave first, so the handle is not closed
        # under it. A send that is stuck on a peer that does not read ends this wait.
        got_recv = self._recv_lock.acquire(timeout=CLOSE_WAIT_S)
        got_send = self._send_lock.acquire(timeout=CLOSE_WAIT_S)
        try:
            self._close_connection()
        finally:
            if got_send:
                self._send_lock.release()
            if got_recv:
                self._recv_lock.release()

    def _close_connection(self) -> None:
        try:
            self._conn.close()
        except OSError:
            _log.debug("closing %s failed", self.name, exc_info=True)

    def __enter__(self) -> Wire:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
