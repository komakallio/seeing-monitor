"""The stream channel: messages, flow control, and what happens when the peer goes."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from seeingmon.services.ipc.client import connect_channel
from seeingmon.services.ipc.errors import (
    IpcClosedError,
    IpcProtocolError,
    StreamCreditError,
)
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.ipc.stream import (
    HEADER_SIZE,
    StreamKind,
    StreamReceiver,
    StreamSender,
    StreamService,
    StreamWindow,
    connect_stream,
    decode_message,
    encode_message,
)
from seeingmon.services.ipc.wire import Wire

from .conftest import wait_until


def take_all(receiver: StreamReceiver, into: list[int]) -> None:
    """Take messages and note the first byte of each, until the receiver raises."""
    while True:
        message = receiver.recv(5.0)
        assert message is not None
        into.append(message.payload[0])


def pump_until_it_fails(sender: StreamSender) -> None:
    """Read acknowledgements until the sender raises. The test fails by timeout if it never does."""
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        sender.pump(0.1)


class TestMessages:
    @given(
        kind=st.sampled_from(list(StreamKind)),
        seq=st.integers(0, 2**64 - 1),
        tag=st.integers(0, 2**32 - 1),
        payload=st.binary(max_size=2000),
    )
    def test_round_trip(self, kind: StreamKind, seq: int, tag: int, payload: bytes) -> None:
        message = decode_message(encode_message(kind, seq, tag, payload))
        assert (message.kind, message.seq, message.tag) == (kind, seq, tag)
        assert bytes(message.payload) == payload

    def test_the_header_is_twenty_bytes(self) -> None:
        assert HEADER_SIZE == 20
        assert len(encode_message(StreamKind.ACK, 1, 0)) == 20

    def test_the_layout_is_fixed(self) -> None:
        raw = encode_message(StreamKind.EVENT, 0x0102030405060708, 0x0A0B0C0D, b"xyz")
        assert raw == (
            b"SMSP"
            + bytes([2, 0, 0, 0])
            + bytes.fromhex("0807060504030201")
            + bytes.fromhex("0d0c0b0a")
            + b"xyz"
        )

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"SMSP" + bytes(15), id="short"),
            pytest.param(b"XXXX" + bytes([1]) + bytes(15), id="magic"),
            pytest.param(b"SMSP" + bytes([9]) + bytes(15), id="kind"),
            pytest.param(b"SMSP" + bytes([1, 1]) + bytes(14), id="reserved"),
        ],
    )
    def test_malformed_messages_are_protocol_errors(self, raw: bytes) -> None:
        with pytest.raises(IpcProtocolError):
            decode_message(raw)

    def test_fields_that_do_not_fit_are_refused_when_encoding(self) -> None:
        with pytest.raises(ValueError, match="fit"):
            encode_message(StreamKind.DATA, 1, 2**32)
        with pytest.raises(ValueError, match="fit"):
            encode_message(StreamKind.DATA, -1, 0)

    def test_the_payload_is_a_read_only_view(self) -> None:
        message = decode_message(encode_message(StreamKind.DATA, 1, 0, b"abc"))
        assert message.payload.readonly
        assert bytes(message.payload) == b"abc"


class TestWindow:
    def test_a_window_needs_room_for_a_message(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            StreamWindow(0, 10)
        with pytest.raises(ValueError, match="at least one"):
            StreamWindow(1, 0)

    def test_the_limit_applies_to_each_dimension(self) -> None:
        assert StreamWindow(100, 10).limited_by(StreamWindow(10, 100)) == StreamWindow(10, 10)

    def test_json_round_trip_and_checks(self) -> None:
        window = StreamWindow(5, 600)
        assert StreamWindow.from_json(window.to_json()) == window
        with pytest.raises(ValueError, match="window"):
            StreamWindow.from_json({"messages": 0, "bytes": 1})
        with pytest.raises(ValueError, match="window"):
            StreamWindow.from_json({"messages": 1})


@dataclass
class Link:
    """A connected sender and receiver, with what the server saw."""

    sender: StreamSender
    receiver: StreamReceiver
    params: Mapping[str, Any]
    granted: StreamWindow


@dataclass
class Collector:
    senders: list[StreamSender] = field(default_factory=list)
    params: list[Mapping[str, Any]] = field(default_factory=list)
    arrived: threading.Event = field(default_factory=threading.Event)

    def on_sender(self, sender: StreamSender, params: Mapping[str, Any]) -> None:
        self.senders.append(sender)
        self.params.append(params)
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
    def open_link(
        window: StreamWindow | None = None,
        *,
        max_window: StreamWindow | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Link:
        collector = Collector()
        service = StreamService(collector.on_sender, max_window=max_window)
        server = start_server({"frames": service})
        receiver, reply = connect_stream(
            server.endpoint, key, params, channel="frames", window=window
        )
        receivers.append(receiver)
        assert collector.arrived.wait(10.0)
        granted = StreamWindow.from_json(reply["window"])
        return Link(collector.senders[0], receiver, collector.params[0], granted)

    return open_link


class TestTransfer:
    def test_messages_arrive_in_order_with_their_tags(self, open_link: Callable[..., Link]) -> None:
        link = open_link()
        for number in range(1, 51):
            assert link.sender.send(f"frame {number}".encode(), tag=number % 3) == number
        for number in range(1, 51):
            message = link.receiver.recv(5.0)
            assert message is not None
            assert (message.seq, message.tag, message.kind) == (number, number % 3, StreamKind.DATA)
            assert bytes(message.payload) == f"frame {number}".encode()

    def test_events_share_the_sequence_with_data(self, open_link: Callable[..., Link]) -> None:
        link = open_link()
        link.sender.send(b"a")
        link.sender.send(b'{"error":"x"}', kind=StreamKind.EVENT)
        link.sender.send(b"b")
        kinds = [link.receiver.recv(5.0) for _ in range(3)]
        assert [m.kind for m in kinds if m is not None] == [
            StreamKind.DATA,
            StreamKind.EVENT,
            StreamKind.DATA,
        ]
        assert [m.seq for m in kinds if m is not None] == [1, 2, 3]

    def test_the_hello_parameters_reach_the_service(self, open_link: Callable[..., Link]) -> None:
        link = open_link(params={"session": "s-7"})
        assert link.params["session"] == "s-7"
        assert link.params["window"] == StreamWindow().to_json()

    def test_recv_returns_none_when_idle_and_wakes_on_arrival(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link()
        started = time.monotonic()
        assert link.receiver.recv(0.2) is None
        assert time.monotonic() - started >= 0.1
        timer = threading.Timer(0.2, lambda: link.sender.send(b"late"))
        timer.start()
        try:
            message = link.receiver.recv(5.0)
        finally:
            timer.join()
        assert message is not None
        assert bytes(message.payload) == b"late"

    def test_a_large_payload_survives(self, open_link: Callable[..., Link]) -> None:
        link = open_link(StreamWindow(4, 64 * 1024 * 1024))
        payload = bytes(range(256)) * (4 * 1024 * 1024 // 256)
        link.sender.send(payload, tag=9)
        message = link.receiver.recv(30.0)
        assert message is not None
        assert bytes(message.payload) == payload

    def test_a_fast_producer_and_a_slow_consumer_lose_nothing(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(8, 1024 * 1024))
        count = 600
        errors: list[BaseException] = []

        def produce() -> None:
            try:
                for number in range(1, count + 1):
                    assert link.sender.wait_credit(100, 30.0)
                    link.sender.send(number.to_bytes(4, "big") + b"x" * 96)
            except BaseException as error:
                errors.append(error)

        producer = threading.Thread(target=produce)
        producer.start()
        received = []
        for _ in range(count):
            message = link.receiver.recv(30.0)
            assert message is not None
            received.append(int.from_bytes(message.payload[:4], "big"))
            if len(received) % 50 == 0:
                time.sleep(0.01)  # the consumer is slower than the producer
        producer.join(30.0)
        assert errors == []
        assert received == list(range(1, count + 1))
        assert link.receiver.pending == 0


class TestFlowControl:
    def test_the_message_window_stops_the_sender_until_the_receiver_takes_one(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(3, 1 << 20))
        assert link.granted == StreamWindow(3, 1 << 20)
        for _ in range(3):
            assert link.sender.has_credit(10)
            link.sender.send(b"0123456789")
        assert not link.sender.has_credit(10)
        with pytest.raises(StreamCreditError):
            link.sender.send(b"x")
        assert link.sender.in_flight == 3
        assert link.receiver.recv(5.0) is not None  # taking one acknowledges it
        assert link.sender.wait_credit(10, 5.0)
        assert link.sender.in_flight == 2
        assert link.sender.acked == 1
        link.sender.send(b"x")

    def test_the_byte_window_counts_what_is_in_flight(self, open_link: Callable[..., Link]) -> None:
        link = open_link(StreamWindow(100, 1000))
        link.sender.send(b"a" * 600)
        assert not link.sender.has_credit(600)
        assert link.sender.has_credit(400)
        link.sender.send(b"b" * 400)
        assert not link.sender.has_credit(1)

    def test_a_message_larger_than_the_window_goes_out_alone(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(10, 100))
        assert link.sender.has_credit(5000)
        link.sender.send(b"z" * 5000)
        assert not link.sender.has_credit(1)
        message = link.receiver.recv(5.0)
        assert message is not None
        assert len(message.payload) == 5000
        assert link.sender.wait_credit(1, 5.0)

    def test_wait_credit_times_out_when_the_receiver_does_not_consume(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(1, 1000))
        link.sender.send(b"x")
        started = time.monotonic()
        assert not link.sender.wait_credit(1, 0.3)
        assert 0.15 <= time.monotonic() - started < 20.0

    def test_the_service_limits_the_window_that_the_receiver_asks_for(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link(StreamWindow(1000, 1 << 30), max_window=StreamWindow(5, 1 << 20))
        assert link.granted == StreamWindow(5, 1 << 20)
        assert link.sender.window == link.granted
        assert link.receiver.window == link.granted


class TestPeerGoes:
    def test_the_receiver_reads_what_arrived_before_it_sees_the_close(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link()
        for number in range(5):
            link.sender.send(bytes([number]))
        assert wait_until(
            lambda: link.receiver.pending == 5
        )  # all five are in the receiver's queue
        link.sender.close()
        received: list[int] = []
        with pytest.raises(IpcClosedError):
            take_all(link.receiver, received)
        assert received == [0, 1, 2, 3, 4]
        assert link.receiver.closed

    def test_the_sender_notices_a_receiver_that_left(self, open_link: Callable[..., Link]) -> None:
        link = open_link()
        link.receiver.close()
        with pytest.raises(IpcClosedError):
            pump_until_it_fails(link.sender)
        with pytest.raises(IpcClosedError):
            link.sender.has_credit(1)

    def test_a_blocked_recv_wakes_when_the_receiver_closes(
        self, open_link: Callable[..., Link]
    ) -> None:
        link = open_link()
        outcome: list[BaseException] = []

        def wait() -> None:
            try:
                link.receiver.recv(None)
            except IpcClosedError as error:
                outcome.append(error)

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.2)
        link.receiver.close()
        thread.join(5.0)
        assert not thread.is_alive()
        assert len(outcome) == 1

    def test_closing_twice_is_fine(self, open_link: Callable[..., Link]) -> None:
        link = open_link()
        link.receiver.close()
        link.receiver.close()
        link.sender.close()
        link.sender.close()


class TestProtocolViolations:
    """A raw peer breaks the rules. The honest end closes, and nothing else happens."""

    def raw_pair(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> tuple[Wire, StreamSender]:
        collector = Collector()
        server = start_server({"frames": StreamService(collector.on_sender)})
        wire, _ = connect_channel(
            server.endpoint,
            key,
            "frames",
            {"window": StreamWindow(2, 1 << 20).to_json()},
            max_message_bytes=1 << 20,
        )
        assert collector.arrived.wait(10.0)
        return wire, collector.senders[0]

    def test_a_receiver_that_acknowledges_what_was_not_sent_is_cut_off(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        wire, sender = self.raw_pair(start_server, key)
        with wire:
            sender.send(b"one")
            wire.send(encode_message(StreamKind.ACK, 5, 0))
            with pytest.raises(IpcProtocolError):
                pump_until_it_fails(sender)
            assert sender.closed

    def test_a_receiver_that_sends_data_is_cut_off(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        wire, sender = self.raw_pair(start_server, key)
        with wire:
            wire.send(encode_message(StreamKind.DATA, 1, 0, b"x"))
            with pytest.raises(IpcProtocolError):
                pump_until_it_fails(sender)

    def test_garbage_from_the_receiver_is_cut_off(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        wire, sender = self.raw_pair(start_server, key)
        with wire:
            wire.send(b"\x80\x04 not a stream message")
            with pytest.raises(IpcProtocolError):
                pump_until_it_fails(sender)

    def serve_raw(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey, raw_messages: list[bytes]
    ) -> StreamReceiver:
        """A server whose handler sends the given raw messages, ignoring flow control."""

        def on_sender(sender: StreamSender, params: Mapping[str, Any]) -> None:
            for raw in raw_messages:
                sender._wire.send(raw)  # bypass the credit check on purpose

        server = start_server({"frames": StreamService(on_sender)})
        receiver, _ = connect_stream(
            server.endpoint, key, channel="frames", window=StreamWindow(2, 1 << 20)
        )
        return receiver

    @pytest.mark.parametrize(
        "messages",
        [
            pytest.param([encode_message(StreamKind.DATA, 2, 0, b"x")], id="skipped-sequence"),
            pytest.param([encode_message(StreamKind.DATA, 1, 0, b"x")] * 2, id="repeated-sequence"),
            pytest.param([encode_message(StreamKind.ACK, 1, 0)], id="ack-from-sender"),
            pytest.param([b"junk"], id="junk"),
            pytest.param(
                [encode_message(StreamKind.DATA, number, 0, b"x") for number in (1, 2, 3)],
                id="beyond-the-window",
            ),
        ],
    )
    def test_a_sender_that_breaks_the_rules_closes_the_receiver(
        self,
        start_server: Callable[..., IpcServer],
        key: ConnectionKey,
        receivers: list[StreamReceiver],
        messages: list[bytes],
    ) -> None:
        receiver = self.serve_raw(start_server, key, messages)
        receivers.append(receiver)
        assert wait_until(lambda: receiver.closed)

    def test_a_service_can_refuse_a_receiver(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        def validate(params: Mapping[str, Any]) -> Mapping[str, Any] | None:
            if params.get("session") != "good":
                raise IpcProtocolError("unknown session")
            return {"hello": "you"}

        server = start_server(
            {"frames": StreamService(lambda sender, params: None, validate=validate)}
        )
        with pytest.raises(IpcProtocolError, match="unknown session"):
            connect_stream(server.endpoint, key, {"session": "bad"}, channel="frames")
        receiver, reply = connect_stream(
            server.endpoint, key, {"session": "good"}, channel="frames"
        )
        receiver.close()
        assert reply["hello"] == "you"

    def test_a_hello_without_a_window_is_refused(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        server = start_server({"frames": StreamService(lambda sender, params: None)})
        with pytest.raises(IpcProtocolError, match="window"):
            connect_channel(server.endpoint, key, "frames", {})
