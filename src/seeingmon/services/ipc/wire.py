"""One connection as a pipe of byte messages.

A `Wire` wraps a `multiprocessing.connection` connection and moves byte messages over it. It never
calls `send` or `recv` of the connection, which pickle, so a peer cannot make this process load an
object. A message is bytes, with a length before it.

**Two ways to move the bytes.** On Windows, and for any connection that is not a plain
`multiprocessing.connection.Connection`, the wire calls `send_bytes`, `recv_bytes`, and `poll`.
On Linux and other POSIX systems it moves the same bytes itself on the file descriptor of the
connection, in the format of `send_bytes`: a four-byte length in network order, and then the
message. The reason is cost. For a message of more than 16 KB `send_bytes` makes two system calls,
and `poll` builds a selector each time, and a stream of 100 frames a second pays for that work 100
times a second. The wire writes the length and the message with one call (`writev`), it waits on a
poll object that it keeps, and it reads a message with one or two calls into one buffer.

**Threads.** Any thread may send: a lock keeps messages whole. One thread at a time should
receive. A receive waits in slices of 0.1 s, so `close` from another thread takes effect within
a slice, on every platform. (Closing a socket does not wake a blocked read on Linux.)

**Limits.** Every receive has a maximum message size, and an oversized message closes the
wire. A connection that has not authenticated yet gets a few hundred bytes.
"""

from __future__ import annotations

import logging
import math
import os
import select
import struct
import sys
import threading
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any, Protocol

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.services.ipc.errors import IpcClosedError, IpcProtocolError

POLL_SLICE_S = 0.1
CLOSE_WAIT_S = 1.0
LENGTH_BYTES = 4
READ_AT_ONCE_BYTES = 256 * 1024  # a larger message is read into its own buffer from the start

_LENGTH = struct.Struct("!i")  # the length that `multiprocessing.connection` puts before a message
# The calls of the POSIX way to move the bytes. They are `None` where the system lacks them.
_WRITEV: Callable[[int, Sequence[Any]], int] | None = getattr(os, "writev", None)
_READV: Callable[[int, Sequence[Any]], int] | None = getattr(os, "readv", None)
_POLL: Callable[[], Any] | None = getattr(select, "poll", None)
_POLLIN: int = getattr(select, "POLLIN", 1)
_POSIX_IO = sys.platform != "win32" and None not in (_WRITEV, _READV, _POLL)

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
        self._fd = -1  # the descriptor, for the POSIX way to move bytes
        self._poller: Any = None
        if _POSIX_IO and _POLL is not None and isinstance(connection, Connection):
            self._fd = connection.fileno()
            self._poller = _POLL()
            self._poller.register(self._fd, _POLLIN)

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
                if self._fd >= 0:
                    self._write_message(data)
                else:
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
                    if self._readable(slice_s):
                        data = self._read_message(limit)
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

    # --- The two ways to move the bytes ----------------------------------------------------

    def _readable(self, timeout_s: float) -> bool:
        """Whether a message, or the end of the connection, is there within `timeout_s`."""
        poller = self._poller
        if poller is None:
            return self._conn.poll(timeout_s)
        return bool(poller.poll(math.ceil(timeout_s * 1000)))

    def _read_message(self, limit: int) -> bytes:
        """Read one whole message. Raises `ValueError("bad message length")` over `limit`."""
        if self._fd < 0:
            return self._conn.recv_bytes(limit)
        fd = self._fd
        header = os.read(fd, LENGTH_BYTES)
        if len(header) < LENGTH_BYTES:
            header = self._read_rest(header, LENGTH_BYTES)
        (size,) = _LENGTH.unpack(header)
        if size < 0 or size > limit:
            raise ValueError("bad message length")
        if size == 0:
            return b""
        if size > READ_AT_ONCE_BYTES:
            return self._read_rest(b"", size)
        data = os.read(fd, size)
        if len(data) == size:
            return data  # the whole message came at once, which is the usual case
        if not data:
            raise OSError("got end of file during message")
        return self._read_rest(data, size)

    def _read_rest(self, first: bytes, size: int) -> bytes:
        """Read the rest of a message of `size` bytes, after the first `first` bytes of it."""
        if not first and size == LENGTH_BYTES:
            raise EOFError  # the connection ended between two messages
        assert _READV is not None
        buffer = bytearray(size)
        view = memoryview(buffer)
        view[: len(first)] = first
        got = len(first)
        while got < size:
            count = _READV(self._fd, [view[got:]])
            if count == 0:
                raise OSError("got end of file during message")
            got += count
        return bytes(buffer)

    def _write_message(self, data: bytes | bytearray | memoryview) -> None:
        """Write the length and the message with one call, and finish a call that fell short."""
        body = memoryview(data)
        if body.format != "B" or body.ndim != 1:
            body = body.cast("B")
        assert _WRITEV is not None
        header = _LENGTH.pack(body.nbytes)
        total = LENGTH_BYTES + body.nbytes
        sent = _WRITEV(self._fd, (header, body))
        while sent < total:
            if sent < LENGTH_BYTES:
                sent += _WRITEV(self._fd, (header[sent:], body))
            else:
                sent += _WRITEV(self._fd, (body[sent - LENGTH_BYTES :],))

    # --- Closing ---------------------------------------------------------------------------

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
