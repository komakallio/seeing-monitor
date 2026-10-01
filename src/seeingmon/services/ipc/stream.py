"""A binary stream channel with sequence numbers and flow control.

A stream carries opaque byte payloads from a sender to a receiver, such as the frames that
`acquire` sends to `core` (see `seeingmon.frames.encode_frame`) or the preview images that
`core` sends to `web`. Other messages never share the connection: the channel has its own
connection, and its messages are of three kinds.

**Messages.** Each message has a 20-byte header and then the payload. All numbers are little
endian.

    offset  size  field
    0       4     magic `SMSP`
    4       1     kind: 1 data, 2 event, 3 acknowledgement
    5       3     reserved, zero
    8       8     sequence number
    16      4     tag, free for the application (`acquire` puts the capture epoch here)

Data and event messages travel from the sender to the receiver and carry consecutive sequence
numbers that start at 1. An event is a small JSON message that must keep its place in the
order of the data, such as a camera error. The receiver answers with an acknowledgement: its
sequence number is the last message that the application has taken, and it has no payload.

**Flow control.** The receiver announces a window of messages and bytes. The sender keeps at most
that much in flight, which means sent and not yet acknowledged, and it sends the next message only
when `has_credit` says that the window has room. A message that alone exceeds the byte window
goes out when nothing else is in flight. So a slow consumer never makes the sender buffer without
limit: the sender stops, and the application above it decides what to drop. The receiver has a
reader thread that takes every message off the connection at once and queues it, so the
connection never blocks the sender for long, and the receiver's queue never exceeds the window.

**Cost per message.** A stream of 100 frames a second pays for every message, so the layer keeps
the work of each one small. The sender reads acknowledgements only when the window has no room and
after every `PUMP_EVERY_SENDS` messages, and not before every send. A receiver may acknowledge in
batches (`ack_every`): it acknowledges the last message that it took after that many messages, or
after that many bytes, or when it has waited `ack_idle_s` for the next message, so a quiet stream
never leaves the sender without credit. `send_into` builds a message in one buffer, with the
payload written in place, so the bytes are copied once.

**Threads.** One thread uses a `StreamSender`. It both sends and reads acknowledgements, so a
sender needs no thread of its own and notices a peer that has gone. A `StreamReceiver` has a
reader thread, and one thread at a time calls `recv`.
"""

from __future__ import annotations

import contextlib
import queue
import struct
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.services.ipc.client import connect_channel
from seeingmon.services.ipc.codec import CodecError, as_mapping, get_int
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import (
    IpcClosedError,
    IpcError,
    IpcProtocolError,
    StreamCreditError,
)
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import Accepted
from seeingmon.services.ipc.wire import Wire

STREAM_MAGIC = b"SMSP"
_HEADER = struct.Struct("<4sBBHQI")
HEADER_SIZE = _HEADER.size
MAX_ACK_BYTES = 64
MAX_ACKS_PER_PUMP = 1000
PUMP_EVERY_SENDS = 16
DEFAULT_ACK_IDLE_S = 0.1
NO_BYTE_LIMIT = 2**62
DEFAULT_STREAM_BYTES = 128 * 1024 * 1024
MAX_TAG = 2**32 - 1

_REAL_CLOCK = SystemClock()


class StreamKind(IntEnum):
    """What a stream message is."""

    DATA = 1
    EVENT = 2
    ACK = 3


@dataclass(frozen=True, slots=True)
class StreamWindow:
    """How much a receiver accepts before it has consumed the first messages."""

    messages: int = 64
    bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.messages < 1 or self.bytes < 1:
            raise ValueError("a window has at least one message and one byte")

    def limited_by(self, limit: StreamWindow) -> StreamWindow:
        """The smaller of the two windows, in each dimension."""
        return StreamWindow(min(self.messages, limit.messages), min(self.bytes, limit.bytes))

    def to_json(self) -> dict[str, int]:
        """The window as a JSON object."""
        return {"messages": self.messages, "bytes": self.bytes}

    @classmethod
    def from_json(cls, value: Any, what: str = "window") -> StreamWindow:
        """The inverse of `to_json`. Raises `CodecError`."""
        data = as_mapping(value, what)
        messages = get_int(data, "messages", what)
        size = get_int(data, "bytes", what)
        if messages < 1 or size < 1:
            raise CodecError(f"{what} must have at least one message and one byte")
        return cls(messages, size)


@dataclass(frozen=True, slots=True)
class StreamMessage:
    """One message of a stream. `payload` is a read-only view of the received bytes."""

    kind: StreamKind
    seq: int
    tag: int
    payload: memoryview


def encode_message(
    kind: StreamKind, seq: int, tag: int, payload: bytes | bytearray | memoryview = b""
) -> bytes:
    """Build the bytes of a message: the header, then the payload."""
    if not 0 <= tag <= MAX_TAG or seq < 0:
        raise ValueError("seq and tag must fit their fields")
    return _HEADER.pack(STREAM_MAGIC, int(kind), 0, 0, seq, tag) + bytes(payload)


def decode_message(raw: bytes) -> StreamMessage:
    """Parse the bytes of a message. Raises `IpcProtocolError` for anything malformed."""
    if len(raw) < HEADER_SIZE:
        raise IpcProtocolError("a stream message is shorter than its header")
    magic, kind, reserved, reserved16, seq, tag = _HEADER.unpack_from(raw)
    if magic != STREAM_MAGIC:
        raise IpcProtocolError("a stream message has a bad magic")
    if reserved or reserved16:
        raise IpcProtocolError("a stream message uses a reserved field")
    try:
        parsed = StreamKind(kind)
    except ValueError:
        raise IpcProtocolError("a stream message has an unknown kind") from None
    return StreamMessage(parsed, seq, tag, memoryview(raw)[HEADER_SIZE:])


class StreamSender:
    """The sending end. Take the credit, send, and read the acknowledgements from one thread."""

    def __init__(self, wire: Wire, window: StreamWindow, *, name: str = "stream") -> None:
        self._wire = wire
        self._window = window
        self.name = name
        self._next_seq = 1
        self._acked = 0
        self._in_flight: deque[tuple[int, int]] = deque()
        self._in_flight_bytes = 0
        self._since_pump = 0
        self.messages_sent = 0
        self.bytes_sent = 0

    @property
    def window(self) -> StreamWindow:
        """The window that the receiver announced."""
        return self._window

    @property
    def closed(self) -> bool:
        """Whether the connection is closed."""
        return self._wire.closed

    @property
    def in_flight(self) -> int:
        """Messages that were sent and that the receiver has not acknowledged."""
        return len(self._in_flight)

    @property
    def acked(self) -> int:
        """The sequence number of the last message that the receiver took."""
        return self._acked

    def pump(self, timeout_s: float = 0.0) -> None:
        """Read the acknowledgements that arrived, and wait up to `timeout_s` for the first.

        Raises `IpcClosedError` when the receiver has gone away, and `IpcProtocolError` when it
        acknowledges something that was never sent.
        """
        self._since_pump = 0
        try:
            raw = self._wire.recv(timeout_s, max_bytes=MAX_ACK_BYTES)
            handled = 0
            while raw is not None and handled < MAX_ACKS_PER_PUMP:
                self._on_ack(raw)
                handled += 1
                raw = self._wire.recv(0.0, max_bytes=MAX_ACK_BYTES)
        except IpcProtocolError:
            self._wire.close("the receiver sent an unreadable message")
            raise

    def _on_ack(self, raw: bytes) -> None:
        message = decode_message(raw)
        if message.kind is not StreamKind.ACK or len(message.payload):
            raise IpcProtocolError("the receiver sent something other than an acknowledgement")
        if not self._acked <= message.seq < self._next_seq:
            raise IpcProtocolError("the receiver acknowledged a message that was never sent")
        self._acked = message.seq
        while self._in_flight and self._in_flight[0][0] <= message.seq:
            _, size = self._in_flight.popleft()
            self._in_flight_bytes -= size

    def has_credit(self, nbytes: int) -> bool:
        """Whether a message of `nbytes` fits in the window now.

        The answer comes from the acknowledgements that the sender has read. When the window looks
        full, the call reads the acknowledgements that wait, and answers again. Raises
        `IpcClosedError` when the connection is closed.
        """
        if self._wire.closed:
            raise IpcClosedError(self._wire.reason)
        if self._fits(nbytes):
            return True
        self.pump(0.0)
        return self._fits(nbytes)

    def room(self) -> tuple[int, int]:
        """The room in the window as the sender knows it: messages, and bytes of payload.

        A message that is larger than the bytes still goes out when nothing is in flight, so the
        bytes are `NO_BYTE_LIMIT` then. The numbers come from the acknowledgements that the
        sender has read. The call reads no more. Use it to choose how much to put in one message.
        """
        count = len(self._in_flight)
        if count == 0:
            return self._window.messages, NO_BYTE_LIMIT
        return self._window.messages - count, max(self._window.bytes - self._in_flight_bytes, 0)

    def _fits(self, nbytes: int) -> bool:
        count = len(self._in_flight)
        if count >= self._window.messages:
            return False
        return count == 0 or self._in_flight_bytes + nbytes <= self._window.bytes

    def wait_credit(self, nbytes: int, timeout_s: float, *, clock: Clock | None = None) -> bool:
        """Wait up to `timeout_s` for room for a message of `nbytes`. Returns whether there is."""
        clock = _REAL_CLOCK if clock is None else clock
        started_ns = clock.monotonic_ns()
        while True:
            if self.has_credit(nbytes):
                return True
            elapsed_s = (clock.monotonic_ns() - started_ns) / NS_PER_S
            if elapsed_s >= timeout_s:
                return False
            self.pump(min(0.05, timeout_s - elapsed_s))

    def send(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        tag: int = 0,
        kind: StreamKind = StreamKind.DATA,
    ) -> int:
        """Send a message and return its sequence number.

        Raises `StreamCreditError` when the window has no room (check `has_credit` first), and
        `IpcClosedError` when the connection is gone.
        """
        if kind is StreamKind.ACK:
            raise ValueError("a sender sends data and events")
        if not self.has_credit(len(payload)):
            raise StreamCreditError("the receiver's window is full")
        seq = self._next_seq
        self._wire.send(encode_message(kind, seq, tag, payload))
        return self._sent(seq, len(payload))

    def send_into(
        self,
        size: int,
        fill: Callable[[memoryview], None],
        *,
        tag: int = 0,
        kind: StreamKind = StreamKind.DATA,
    ) -> int:
        """Send a message of `size` payload bytes that `fill` writes in place. Returns its number.

        `fill` receives a writable view of the payload part of the message, and it must write all
        `size` bytes. The message then goes out from one buffer, so the payload is copied once.
        The call has the errors of `send`.
        """
        if kind is StreamKind.ACK:
            raise ValueError("a sender sends data and events")
        if not 0 <= tag <= MAX_TAG:
            raise ValueError("seq and tag must fit their fields")
        if not self.has_credit(size):
            raise StreamCreditError("the receiver's window is full")
        seq = self._next_seq
        buffer = bytearray(HEADER_SIZE + size)
        _HEADER.pack_into(buffer, 0, STREAM_MAGIC, int(kind), 0, 0, seq, tag)
        fill(memoryview(buffer)[HEADER_SIZE:])
        self._wire.send(buffer)
        return self._sent(seq, size)

    def _sent(self, seq: int, size: int) -> int:
        """Book a message that went out. Reads the acknowledgements now and then."""
        self._next_seq += 1
        self._in_flight.append((seq, size))
        self._in_flight_bytes += size
        self.messages_sent += 1
        self.bytes_sent += size
        self._since_pump += 1
        if self._since_pump >= PUMP_EVERY_SENDS:
            self.pump(0.0)  # also notices a receiver that has gone away
        return seq

    def close(self, reason: str = "the sender closed the stream") -> None:
        """Close the connection. The receiver reads what it already has, and then sees the close.

        The sender first reads the acknowledgements that wait for it. A socket that closes with
        unread data resets the connection, and a reset makes the peer drop what it has not read.
        """
        with contextlib.suppress(IpcError):
            self.pump(0.0)
        self._wire.close(reason)


class StreamReceiver:
    """The receiving end. A reader thread queues the messages, and `recv` takes them."""

    def __init__(
        self,
        wire: Wire,
        window: StreamWindow,
        *,
        name: str = "stream",
        clock: Clock | None = None,
        ack_every: int = 1,
        ack_idle_s: float = DEFAULT_ACK_IDLE_S,
    ) -> None:
        if ack_every < 1 or ack_idle_s <= 0:
            raise ValueError("ack_every is at least 1, and ack_idle_s is positive")
        self._wire = wire
        self._window = window
        self.name = name
        self._clock = _REAL_CLOCK if clock is None else clock
        # The reader puts messages in a queue, and `None` after the last one. A `SimpleQueue` is
        # cheaper than a condition: the reader pays one call per message and the consumer one.
        self._queue: queue.SimpleQueue[StreamMessage | None] = queue.SimpleQueue()
        self._state_lock = threading.Lock()
        self._expected = 1
        self._finished = False  # the end marker is in the queue
        self._ended = False  # `recv` took the end marker
        self._reason = "the stream is closed"
        self._ack_every = ack_every
        self._ack_idle_s = ack_idle_s
        self._ack_bytes = max(1, window.bytes // 4)
        self._unacked = 0  # messages that `recv` returned and that no acknowledgement covers
        self._unacked_bytes = 0
        self._last_taken = 0
        self.messages_received = 0
        self.bytes_received = 0
        self.acks_sent = 0
        self._reader = threading.Thread(target=self._read_loop, name=f"{name}-reader", daemon=True)
        self._reader.start()

    @property
    def window(self) -> StreamWindow:
        """The window that this receiver granted."""
        return self._window

    @property
    def closed(self) -> bool:
        """Whether the connection closed. Messages that arrived before may still be queued."""
        return self._finished

    @property
    def pending(self) -> int:
        """Messages that arrived and that `recv` has not returned."""
        marker = 1 if self._finished and not self._ended else 0
        return max(self._queue.qsize() - marker, 0)

    def recv(self, timeout_s: float | None = None) -> StreamMessage | None:
        """Take the next message, or return `None` when `timeout_s` passes without one.

        Messages that arrived before the connection closed are returned first. After the last
        one, the call raises `IpcClosedError`. Taking a message gives the sender credit for
        another: the receiver acknowledges at once (`ack_every=1`), or in batches (see the module
        text), and it always acknowledges before it waits for longer than `ack_idle_s`.
        """
        deadline_ns = (
            None if timeout_s is None else self._clock.monotonic_ns() + round(timeout_s * NS_PER_S)
        )
        while True:
            if self._ended:
                raise IpcClosedError(self._reason)
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                wait_s: float | None = None
                if deadline_ns is not None:
                    wait_s = (deadline_ns - self._clock.monotonic_ns()) / NS_PER_S
                    if wait_s <= 0:
                        self.flush_acks()  # a quiet stream must not leave the sender without credit
                        return None
                idle_wait = bool(self._unacked) and (wait_s is None or wait_s > self._ack_idle_s)
                try:
                    message = self._queue.get(timeout=self._ack_idle_s if idle_wait else wait_s)
                except queue.Empty:
                    if idle_wait:
                        self.flush_acks()
                    continue
            if message is None:  # the end marker: the connection closed after the last message
                self._ended = True
                raise IpcClosedError(self._reason)
            break
        self._unacked += 1
        self._unacked_bytes += len(message.payload) + HEADER_SIZE
        self._last_taken = message.seq
        if self._unacked >= self._ack_every or self._unacked_bytes >= self._ack_bytes:
            self.flush_acks()
        return message

    def flush_acks(self) -> None:
        """Acknowledge the messages that `recv` returned and that no acknowledgement covers yet."""
        if not self._unacked:
            return
        seq = self._last_taken
        self._unacked = 0
        self._unacked_bytes = 0
        with contextlib.suppress(IpcClosedError):  # `recv` reports a closed wire after the queue
            self._wire.send(encode_message(StreamKind.ACK, seq, 0))
            self.acks_sent += 1

    def close(self, reason: str = "the receiver closed the stream") -> None:
        """Close the connection and wake a thread that waits in `recv`."""
        self._wire.close(reason)
        self._finish(reason)
        if self._reader is not threading.current_thread():
            self._reader.join(2.0)

    def _finish(self, reason: str) -> None:
        with self._state_lock:
            if self._finished:
                return
            self._finished = True
            self._reason = reason
            self._queue.put(None)  # wakes a thread that waits in `recv`

    def _read_loop(self) -> None:
        reason = "the sender closed the stream"
        try:
            while True:
                raw = self._wire.recv(None)
                if raw is None:
                    continue
                message = decode_message(raw)
                if message.kind is StreamKind.ACK:
                    raise IpcProtocolError("the sender sent an acknowledgement")
                if message.seq != self._expected:
                    raise IpcProtocolError("the sender skipped or repeated a sequence number")
                if self._queue.qsize() >= self._window.messages:
                    raise IpcProtocolError("the sender exceeded the window")
                self._queue.put(message)
                self._expected += 1
                self.messages_received += 1
                self.bytes_received += len(message.payload)
        except IpcClosedError:
            reason = self._wire.reason
        except IpcError as error:
            reason = str(error)
            self._wire.close(reason)
        finally:
            self._finish(reason)


class StreamService:
    """The server side of a stream: it gives each new connection to the application as a sender.

    Register it as a channel of an `IpcServer`. `on_sender` receives the `StreamSender` and the
    hello parameters when the client is ready. `validate` may refuse a client by raising an
    `IpcError`, and it may return extra fields for the hello reply. The window is the smaller of
    what the client asks and `max_window`.
    """

    def __init__(
        self,
        on_sender: Callable[[StreamSender, Mapping[str, Any]], None],
        *,
        max_window: StreamWindow | None = None,
        max_message_bytes: int = DEFAULT_STREAM_BYTES,
        validate: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
        name: str = "stream",
    ) -> None:
        self._on_sender = on_sender
        self._max_window = max_window or StreamWindow()
        self._max_message_bytes = max_message_bytes
        self._validate = validate
        self._name = name

    def accept(self, wire: Wire, params: Mapping[str, Any]) -> Accepted:
        """Take a connection that asked for this channel. Called by the `IpcServer`."""
        try:
            asked = StreamWindow.from_json(params.get("window"), "hello.window")
        except CodecError as error:
            raise IpcProtocolError(str(error)) from None
        window = asked.limited_by(self._max_window)
        extra = self._validate(params) if self._validate is not None else None
        wire.max_message_bytes = self._max_message_bytes
        sender = StreamSender(wire, window, name=self._name)
        reply = {"window": window.to_json(), **(extra or {})}
        return Accepted(
            reply=reply,
            activate=lambda: self._on_sender(sender, params),
            abort=sender.close,
        )


def connect_stream(
    endpoint: Endpoint,
    key: ConnectionKey,
    params: Mapping[str, Any] | None = None,
    *,
    channel: str,
    window: StreamWindow | None = None,
    connect_timeout_s: float = 5.0,
    handshake_timeout_s: float = 5.0,
    max_message_bytes: int = DEFAULT_STREAM_BYTES,
    clock: Clock | None = None,
    name: str = "stream",
    ack_batch: int = 1,
) -> tuple[StreamReceiver, Mapping[str, Any]]:
    """Connect to a `StreamService` and return the receiver and the hello reply.

    The receiver's window is the one the service granted, which may be smaller than the one
    asked for.
    """
    asked = window or StreamWindow()
    wire, reply = connect_channel(
        endpoint,
        key,
        channel,
        {**(params or {}), "window": asked.to_json()},
        connect_timeout_s=connect_timeout_s,
        handshake_timeout_s=handshake_timeout_s,
        max_message_bytes=max_message_bytes,
        clock=clock,
    )
    wire.name = name
    try:
        granted = StreamWindow.from_json(reply.get("window"), "hello reply.window")
        if granted.messages > asked.messages or granted.bytes > asked.bytes:
            raise IpcProtocolError("the service granted a larger window than the receiver asked")
    except (CodecError, IpcProtocolError):
        wire.close()
        raise
    ack_every = max(1, min(ack_batch, granted.messages // 4))
    return StreamReceiver(wire, granted, name=name, clock=clock, ack_every=ack_every), reply
