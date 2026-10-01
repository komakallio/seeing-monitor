"""The POSIX way to move bytes: a `Wire` on the file descriptor of a connection."""

from __future__ import annotations

import itertools
import os
import struct
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from multiprocessing.connection import Pipe
from typing import Any

import pytest

from seeingmon.services.ipc import wire as wire_module
from seeingmon.services.ipc.errors import IpcClosedError, IpcProtocolError
from seeingmon.services.ipc.wire import Wire

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the wire uses pipes on Windows")

LIMIT = 8 * 1024 * 1024
SIZES = [0, 1, 3, 4, 5, 1000, 16384, 16385, 65536, 262144, 262145, 1 << 20, 3 << 20]


def payload(size: int) -> bytes:
    return bytes((index * 7 + 3) % 251 for index in range(size))


@pytest.fixture
def pair() -> Iterator[tuple[Wire, Any]]:
    """A wire, and the plain connection at the other end of it."""
    mine, theirs = Pipe(duplex=True)
    wire = Wire(mine, max_message_bytes=LIMIT, name="posix")
    yield wire, theirs
    wire.close()
    theirs.close()


def send_all(connection: Any, sizes: Sequence[int]) -> None:
    for size in sizes:
        connection.send_bytes(payload(size))


def send_until_it_fails(wire: Wire) -> None:
    for _ in range(100):  # the first sends may fit in the buffer of the connection
        wire.send(payload(100_000))


def in_thread(call: Any) -> threading.Thread:
    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    return thread


class TestTheDescriptorPath:
    def test_a_plain_connection_gets_the_descriptor_path(self, pair: tuple[Wire, Any]) -> None:
        wire, _ = pair
        assert wire._fd >= 0
        assert wire._poller is not None

    def test_a_connection_that_is_not_plain_keeps_the_old_calls(self) -> None:
        class Fake:
            def __init__(self) -> None:
                self.sent: list[bytes] = []

            def send_bytes(self, buf: Any) -> None:
                self.sent.append(bytes(buf))

            def recv_bytes(self, maxlength: int | None = None) -> bytes:
                return b"hello"

            def poll(self, timeout: float | None = 0.0) -> bool:
                return True

            def close(self) -> None:
                pass

        fake = Fake()
        wire = Wire(fake, max_message_bytes=100)
        assert wire._fd == -1
        wire.send(b"abc")
        assert fake.sent == [b"abc"]
        assert wire.recv(1.0) == b"hello"

    @pytest.mark.parametrize("size", SIZES)
    def test_a_message_of_any_size_arrives_whole(self, pair: tuple[Wire, Any], size: int) -> None:
        wire, theirs = pair
        other = Wire(theirs, max_message_bytes=LIMIT, name="other")
        sender = in_thread(lambda: wire.send(payload(size)))
        assert other.recv(20.0) == payload(size)
        sender.join(20.0)
        assert not sender.is_alive()

    @pytest.mark.parametrize("size", [0, 7, 20000, 300000])
    def test_the_wire_sends_what_recv_bytes_reads(self, pair: tuple[Wire, Any], size: int) -> None:
        wire, theirs = pair
        sender = in_thread(lambda: wire.send(payload(size)))
        assert theirs.recv_bytes() == payload(size)
        sender.join(20.0)

    @pytest.mark.parametrize("size", [0, 7, 20000, 300000])
    def test_the_wire_reads_what_send_bytes_writes(self, pair: tuple[Wire, Any], size: int) -> None:
        wire, theirs = pair
        sender = in_thread(lambda: theirs.send_bytes(payload(size)))
        assert wire.recv(20.0) == payload(size)
        sender.join(20.0)

    def test_messages_keep_their_order_and_their_edges(self, pair: tuple[Wire, Any]) -> None:
        wire, theirs = pair
        sizes = [5, 0, 70000, 1, 40000, 3]
        sender = in_thread(lambda: send_all(theirs, sizes))
        assert [wire.recv(20.0) for _ in sizes] == [payload(size) for size in sizes]
        sender.join(20.0)

    def test_a_message_that_comes_in_pieces_is_put_together(self, pair: tuple[Wire, Any]) -> None:
        wire, theirs = pair
        message = payload(50000)
        frame = struct.pack("!i", len(message)) + message
        cuts = [0, 1, 3, 4, 5, 20000, 20001, len(frame)]

        def dribble() -> None:
            for start, end in itertools.pairwise(cuts):
                os.write(theirs.fileno(), frame[start:end])
                time.sleep(0.01)

        sender = in_thread(dribble)
        assert wire.recv(20.0) == message
        sender.join(20.0)

    def test_a_send_that_falls_short_is_finished(
        self, pair: tuple[Wire, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wire, theirs = pair
        calls: list[int] = []

        def short_writev(fd: int, buffers: Sequence[Any]) -> int:
            data = b"".join(bytes(buffer) for buffer in buffers)
            written = os.write(fd, data[:3])  # at most three bytes, so the header splits too
            calls.append(written)
            return written

        monkeypatch.setattr(wire_module, "_WRITEV", short_writev)
        sender = in_thread(lambda: wire.send(b"0123456789"))
        assert theirs.recv_bytes() == b"0123456789"
        sender.join(20.0)
        assert sum(calls) == 4 + 10
        assert len(calls) == 5

    def test_a_memoryview_of_another_format_is_sent_as_bytes(self, pair: tuple[Wire, Any]) -> None:
        wire, theirs = pair
        view = memoryview(bytearray(range(32))).cast("H")
        wire.send(view)
        assert theirs.recv_bytes() == bytes(range(32))


class TestTheEnds:
    def test_a_receive_times_out_and_returns_nothing(self, pair: tuple[Wire, Any]) -> None:
        wire, _ = pair
        started = time.monotonic()
        assert wire.recv(0.05) is None
        assert time.monotonic() - started >= 0.04
        assert wire.recv(0.0) is None

    def test_the_end_between_two_messages_closes_the_wire(self, pair: tuple[Wire, Any]) -> None:
        wire, theirs = pair
        theirs.send_bytes(b"last")
        theirs.close()
        assert wire.recv(5.0) == b"last"
        with pytest.raises(IpcClosedError):
            wire.recv(5.0)
        assert wire.closed
        assert wire.reason == "the peer closed the connection"

    def test_the_end_inside_a_message_closes_the_wire(self, pair: tuple[Wire, Any]) -> None:
        wire, theirs = pair
        os.write(theirs.fileno(), struct.pack("!i", 100) + b"x" * 40)
        theirs.close()
        with pytest.raises(IpcClosedError):
            wire.recv(5.0)
        assert wire.closed
        assert "broke" in wire.reason

    def test_the_end_inside_the_length_closes_the_wire(self, pair: tuple[Wire, Any]) -> None:
        wire, theirs = pair
        os.write(theirs.fileno(), b"\x00\x00")
        theirs.close()
        with pytest.raises(IpcClosedError):
            wire.recv(5.0)
        assert "broke" in wire.reason

    def test_a_message_over_the_limit_is_a_protocol_error_and_closes_the_wire(
        self, pair: tuple[Wire, Any]
    ) -> None:
        wire, theirs = pair
        theirs.send_bytes(payload(1000))
        with pytest.raises(IpcProtocolError):
            wire.recv(5.0, max_bytes=100)
        assert wire.closed
        assert wire.reason == "the peer sent a message over the size limit"

    def test_a_negative_length_is_a_protocol_error(self, pair: tuple[Wire, Any]) -> None:
        wire, theirs = pair
        os.write(theirs.fileno(), struct.pack("!i", -1) + b"\x00" * 8)
        with pytest.raises(IpcProtocolError):
            wire.recv(5.0)
        assert wire.closed

    def test_a_send_to_a_peer_that_left_closes_the_wire(self, pair: tuple[Wire, Any]) -> None:
        wire, theirs = pair
        theirs.close()
        with pytest.raises(IpcClosedError):
            send_until_it_fails(wire)
        assert wire.closed

    def test_a_message_over_the_limit_is_refused_before_it_is_sent(
        self, pair: tuple[Wire, Any]
    ) -> None:
        wire, _ = pair
        with pytest.raises(IpcProtocolError):
            wire.send(payload(LIMIT + 1))
        assert not wire.closed
