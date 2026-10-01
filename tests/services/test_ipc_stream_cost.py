"""The work of the stream layer for each message: lazy credit, batched acknowledgements, one buffer.

These tests show how the layer keeps its per-message cost small without giving up flow control or
the way that it notices a peer that left. They count what crosses the wire, and they never time
anything.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from seeingmon.frames import (
    FRAME_HEADER_SIZE,
    Frame,
    FrameFlag,
    PixelFormat,
    Roi,
    TimeQuality,
    decode_frame,
    encode_frame,
    encode_frame_into,
    frame_wire_size,
)
from seeingmon.services.ipc.errors import IpcClosedError, StreamCreditError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.ipc.stream import (
    HEADER_SIZE,
    PUMP_EVERY_SENDS,
    StreamKind,
    StreamReceiver,
    StreamSender,
    StreamService,
    StreamWindow,
    connect_stream,
    decode_message,
)

from .conftest import wait_until


@dataclass
class Link:
    sender: StreamSender
    receiver: StreamReceiver


@dataclass
class Collector:
    senders: list[StreamSender] = field(default_factory=list)
    arrived: threading.Event = field(default_factory=threading.Event)

    def on_sender(self, sender: StreamSender, params: Mapping[str, Any]) -> None:
        self.senders.append(sender)
        self.arrived.set()


@pytest.fixture
def receivers() -> Iterator[list[StreamReceiver]]:
    opened: list[StreamReceiver] = []
    yield opened
    for receiver in opened:
        receiver.close()


@pytest.fixture
def open_link(
    start_server: Callable[..., IpcServer],
    key: ConnectionKey,
    receivers: list[StreamReceiver],
) -> Callable[..., Link]:
    def open_link(window: StreamWindow | None = None, *, ack_batch: int = 1) -> Link:
        collector = Collector()
        server = start_server({"frames": StreamService(collector.on_sender)})
        receiver, _ = connect_stream(
            server.endpoint, key, None, channel="frames", window=window, ack_batch=ack_batch
        )
        receivers.append(receiver)
        assert collector.arrived.wait(10.0)
        return Link(collector.senders[0], receiver)

    return open_link


def send_until_it_fails(sender: StreamSender) -> None:
    """Send small messages until the sender raises. The test fails by timeout if it never does."""
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        sender.send(b"x")


def count_reads(sender: StreamSender) -> list[int]:
    """Count the receives that the sender makes on its wire, with the timeout of each."""
    calls: list[int] = []
    wire = sender._wire
    original = wire.recv

    def counting(timeout_s: float | None, *, max_bytes: int | None = None) -> bytes | None:
        calls.append(1)
        return original(timeout_s, max_bytes=max_bytes)

    wire.recv = counting  # type: ignore[method-assign]
    return calls


class TestTheSenderReadsAcknowledgementsLazily:
    def test_a_window_with_room_costs_no_read(self, open_link: Callable[..., Link]) -> None:
        link = open_link(StreamWindow(64, 1 << 20))
        reads = count_reads(link.sender)
        for _ in range(100):
            assert link.sender.has_credit(10)
        assert reads == []

    def test_a_full_window_reads_the_acknowledgements_and_answers_again(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(2, 1 << 20))
        link.sender.send(b"a")
        link.sender.send(b"b")
        reads = count_reads(link.sender)
        assert not link.sender.has_credit(1)  # the window is full, so the sender looks
        assert reads
        assert link.receiver.recv(5.0) is not None
        assert wait_until(lambda: link.sender.has_credit(1), 10.0)

    def test_the_sender_reads_the_acknowledgements_after_every_so_many_messages(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(1000, 1 << 24))
        reads = count_reads(link.sender)
        for number in range(PUMP_EVERY_SENDS - 1):
            link.sender.send(bytes([number]))
        assert reads == []
        link.sender.send(b"x")  # the last of the batch
        assert reads  # one read of whatever the receiver has acknowledged
        assert link.sender.in_flight <= PUMP_EVERY_SENDS

    def test_a_receiver_that_left_is_noticed_without_a_full_window(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(1000, 1 << 24))
        link.receiver.close()
        with pytest.raises(IpcClosedError):
            send_until_it_fails(link.sender)  # the periodic read finds the closed connection
        with pytest.raises(IpcClosedError):
            link.sender.has_credit(1)

    def test_send_still_refuses_a_message_that_has_no_room(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(1, 1 << 20))
        link.sender.send(b"a")
        with pytest.raises(StreamCreditError):
            link.sender.send(b"b")


class TestTheReceiverAcknowledgesInBatches:
    def test_it_acknowledges_after_the_batch_and_not_before(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(64, 1 << 20), ack_batch=4)
        for number in range(1, 4):
            link.sender.send(bytes([number]))
        for _ in range(3):
            assert link.receiver.recv(5.0) is not None
        assert link.receiver.acks_sent == 0  # three messages taken, and none acknowledged yet
        link.sender.send(b"\x04")
        assert link.receiver.recv(5.0) is not None
        assert link.receiver.acks_sent == 1  # the fourth completes the batch
        link.sender.pump(2.0)
        assert link.sender.acked == 4
        assert link.sender.in_flight == 0

    def test_a_batch_that_is_never_completed_is_acknowledged_when_the_stream_goes_quiet(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(64, 1 << 20), ack_batch=8)
        receiver = link.receiver
        receiver._ack_idle_s = 0.05
        link.sender.send(b"a")
        link.sender.send(b"b")
        assert receiver.recv(5.0) is not None
        assert receiver.recv(5.0) is not None
        assert receiver.acks_sent == 0
        assert receiver.recv(0.5) is None  # nothing more comes, so the receiver acknowledges
        assert receiver.acks_sent == 1
        link.sender.pump(2.0)
        assert link.sender.in_flight == 0

    def test_a_wait_for_the_next_message_acknowledges_after_the_idle_time(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(64, 1 << 20), ack_batch=8)
        receiver = link.receiver
        receiver._ack_idle_s = 0.05
        link.sender.send(b"a")
        assert receiver.recv(5.0) is not None
        got: list[Any] = []
        waiter = threading.Thread(target=lambda: got.append(receiver.recv(10.0)))
        waiter.start()
        assert wait_until(lambda: receiver.acks_sent == 1, 10.0)  # while it still waits
        link.sender.send(b"b")
        waiter.join(10.0)
        assert len(got) == 1
        assert got[0] is not None

    def test_large_messages_are_acknowledged_at_once(self, open_link: Callable[..., Link]) -> None:
        link = open_link(StreamWindow(64, 4000), ack_batch=8)  # a quarter of the window is 1000
        link.sender.send(bytes(2000))
        assert link.receiver.recv(5.0) is not None
        assert link.receiver.acks_sent == 1  # 2,000 bytes exceed a quarter of the byte window

    @pytest.mark.parametrize(("messages", "expected"), [(64, 8), (8, 2), (2, 1)])
    def test_the_window_limits_the_batch(
        self, open_link: Callable[..., Link], messages: int, expected: int
    ) -> None:
        link = open_link(StreamWindow(messages, 1 << 20), ack_batch=8)
        assert link.receiver._ack_every == expected

    def test_the_default_acknowledges_every_message(self, open_link: Callable[..., Link]) -> None:
        link = open_link(StreamWindow(64, 1 << 20))
        for _ in range(5):
            link.sender.send(b"x")
            assert link.receiver.recv(5.0) is not None
        assert link.receiver.acks_sent == 5

    def test_a_flow_through_a_small_window_keeps_going(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(8, 1 << 20), ack_batch=8)  # batches of two
        received: list[int] = []

        def consume() -> None:
            for _ in range(200):
                message = link.receiver.recv(10.0)
                assert message is not None
                received.append(message.payload[0])

        consumer = threading.Thread(target=consume)
        consumer.start()
        for number in range(200):
            assert link.sender.wait_credit(1, 20.0)
            link.sender.send(bytes([number % 256]))
        consumer.join(30.0)
        assert received == [number % 256 for number in range(200)]


class TestOneBuffer:
    def test_send_into_makes_the_same_message_as_send(self, open_link: Callable[..., Link]) -> None:
        link = open_link()
        payload = bytes(range(200))

        def fill(view: memoryview) -> None:
            view[:] = payload

        number = link.sender.send_into(len(payload), fill, tag=7)
        link.sender.send(payload, tag=7)
        first = link.receiver.recv(5.0)
        second = link.receiver.recv(5.0)
        assert first is not None
        assert second is not None
        assert number == first.seq
        assert (first.kind, first.tag, bytes(first.payload)) == (
            second.kind,
            second.tag,
            bytes(second.payload),
        )

    def test_the_header_is_the_one_of_encode_message(self, open_link: Callable[..., Link]) -> None:
        seen: list[bytes] = []
        link = open_link()
        wire = link.sender._wire
        original = wire.send

        def spy(data: bytes | bytearray | memoryview) -> None:
            seen.append(bytes(data))
            original(data)

        wire.send = spy  # type: ignore[method-assign]
        link.sender.send_into(3, lambda view: view.__setitem__(slice(None), b"abc"), tag=9)
        message = decode_message(seen[0])
        assert (message.kind, message.seq, message.tag, bytes(message.payload)) == (
            StreamKind.DATA,
            1,
            9,
            b"abc",
        )
        assert len(seen[0]) == HEADER_SIZE + 3

    def test_send_into_checks_the_credit_the_tag_and_the_kind(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(1, 1 << 20))
        link.sender.send_into(1, lambda view: view.__setitem__(0, 1))
        with pytest.raises(StreamCreditError):
            link.sender.send_into(1, lambda view: view.__setitem__(0, 1))
        with pytest.raises(ValueError, match="fit"):
            link.sender.send_into(1, lambda view: None, tag=2**32)
        with pytest.raises(ValueError, match="sender sends"):
            link.sender.send_into(1, lambda view: None, kind=StreamKind.ACK)


def make_frame(dtype: type[np.generic], shape: tuple[int, int], *, fortran: bool = False) -> Frame:
    rng = np.random.default_rng(3)
    data = rng.integers(0, 2000, size=shape).astype(dtype)
    if fortran:
        data = np.asfortranarray(data)
    return Frame(
        data=data,
        stream_id=3,
        seq=17,
        t_arrival_ns=1_700_000_000_000_000_000,
        t_utc_ns=1_700_000_000_000_100_000,
        t_err_ns=1500,
        t_quality=TimeQuality.FITTED,
        dropped_before=2,
        exposure_us=2000,
        gain=120,
        mode="bin1",
        roi=Roi(10, 20, shape[1], shape[0]),
        adc_bits=14,
        temperature_c=12.345,
        flags=FrameFlag.RECOVERED,
    )


class TestEncodeFrameInto:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    @pytest.mark.parametrize("fortran", [False, True])
    def test_it_writes_the_bytes_of_encode_frame(
        self, dtype: type[np.generic], fortran: bool
    ) -> None:
        frame = make_frame(dtype, (24, 40), fortran=fortran)
        out = bytearray(frame_wire_size(frame))
        encode_frame_into(out, frame)
        expected = encode_frame(
            Frame(**{**frame.__dict__, "data": np.ascontiguousarray(frame.data)})
            if hasattr(frame, "__dict__")
            else frame
        )
        assert bytes(out) == expected

    def test_the_decoder_reads_what_it_wrote(self) -> None:
        frame = make_frame(np.uint16, (16, 32))
        out = bytearray(frame_wire_size(frame))
        encode_frame_into(out, frame)
        decoded = decode_frame(out)
        assert np.array_equal(decoded.data, frame.data)
        assert (decoded.seq, decoded.dropped_before, decoded.flags) == (17, 2, FrameFlag.RECOVERED)
        assert decoded.pixel_format is PixelFormat.RAW16

    def test_the_size_is_the_header_and_the_pixels(self) -> None:
        frame = make_frame(np.uint16, (16, 32))
        assert frame_wire_size(frame) == FRAME_HEADER_SIZE + 16 * 32 * 2

    def test_a_buffer_of_the_wrong_size_is_refused(self) -> None:
        frame = make_frame(np.uint8, (8, 8))
        with pytest.raises(ValueError, match="needs"):
            encode_frame_into(bytearray(10), frame)

    def test_it_writes_into_a_view_of_a_larger_buffer(self) -> None:
        frame = make_frame(np.uint8, (8, 8))
        size = frame_wire_size(frame)
        big = bytearray(size + 20)
        encode_frame_into(memoryview(big)[20:], frame)
        assert bytes(big[20:]) == encode_frame(frame)
        assert bytes(big[:20]) == bytes(20)
