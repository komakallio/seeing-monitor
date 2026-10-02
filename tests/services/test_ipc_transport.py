"""The listener, the client, the handshake, and the wire."""

from __future__ import annotations

import os
import pickle
import socket
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import pytest

from seeingmon.services.ipc.client import connect_channel
from seeingmon.services.ipc.codec import decode_json, encode_json
from seeingmon.services.ipc.endpoint import FAMILY_UNIX, Endpoint
from seeingmon.services.ipc.errors import (
    IpcAddressInUseError,
    IpcAuthError,
    IpcClosedError,
    IpcConnectError,
    IpcProtocolError,
)
from seeingmon.services.ipc.handshake import MAGIC
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import Accepted, IpcServer
from seeingmon.services.ipc.wire import Wire

from .conftest import drain, wait_until

PICKLE_RAN: list[int] = []


def _run_me() -> None:
    PICKLE_RAN.append(1)


class Evil:
    """Unpickling an instance calls `_run_me`, so the list shows whether anything unpickled."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (_run_me, ())


class EchoChannel:
    """A channel that sends every message back, and remembers the hello parameters."""

    def __init__(
        self,
        reply: Mapping[str, Any] | None = None,
        *,
        reject: str | None = None,
        max_message_bytes: int = 1 << 20,
    ):
        self.reply = dict(reply or {})
        self.reject = reject
        self.max_message_bytes = max_message_bytes
        self.params: list[Mapping[str, Any]] = []
        self.wires: list[Wire] = []
        self.echoed = 0

    def accept(self, wire: Wire, params: Mapping[str, Any]) -> Accepted:
        if self.reject is not None:
            raise IpcProtocolError(self.reject)
        wire.max_message_bytes = self.max_message_bytes
        self.params.append(dict(params))
        self.wires.append(wire)

        def run() -> None:
            threading.Thread(target=self._echo, args=(wire,), daemon=True).start()

        return Accepted(reply=self.reply, activate=run)

    def _echo(self, wire: Wire) -> None:
        try:
            while True:
                data = wire.recv(0.1)
                if data is not None:
                    wire.send(data)
                    self.echoed += 1
        except (IpcClosedError, IpcProtocolError):
            wire.close()


class TestHandshakeAndHello:
    def test_a_client_with_the_key_reaches_the_channel(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        channel = EchoChannel(reply={"session": "abc"})
        server = start_server({"echo": channel})
        wire, reply = connect_channel(server.endpoint, key, "echo", {"who": "test"})
        with wire:
            assert reply == {"session": "abc"}
            wire.send(b"hello")
            assert wire.recv(5.0) == b"hello"
        assert channel.params == [{"who": "test"}]
        assert server.stats.activated == 1

    def test_the_server_reports_the_real_address(
        self, start_server: Callable[..., IpcServer], endpoint: Endpoint
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        assert server.running
        if isinstance(endpoint.address, tuple):
            assert isinstance(server.endpoint.address, tuple)
            assert server.endpoint.address[1] != 0
        else:
            assert server.endpoint == endpoint

    def test_a_wrong_key_is_refused_and_the_server_keeps_serving(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey, other_key: ConnectionKey
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        with pytest.raises(IpcAuthError):
            connect_channel(server.endpoint, other_key, "echo")
        assert wait_until(lambda: server.stats.auth_failures == 1)
        wire, _ = connect_channel(server.endpoint, key, "echo")
        with wire:
            wire.send(b"still here")
            assert wire.recv(5.0) == b"still here"

    def test_a_peer_that_does_not_know_the_key_cannot_make_the_server_answer(
        self, start_server: Callable[..., IpcServer], other_key: ConnectionKey
    ) -> None:
        channel = EchoChannel()
        server = start_server({"echo": channel})
        with pytest.raises(IpcAuthError):
            connect_channel(server.endpoint, other_key, "echo")
        assert channel.params == []

    def test_an_unknown_channel_is_refused_with_a_message(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        with pytest.raises(IpcProtocolError, match="no channel named 'nope'"):
            connect_channel(server.endpoint, key, "nope")

    def test_a_handler_can_refuse_with_a_reason(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        server = start_server({"echo": EchoChannel(reject="the session is over")})
        with pytest.raises(IpcProtocolError, match="the session is over"):
            connect_channel(server.endpoint, key, "echo")

    def test_nobody_listening_is_a_connect_error_after_the_timeout(
        self, endpoint: Endpoint, key: ConnectionKey, short_dir: Path
    ) -> None:
        if isinstance(endpoint.address, tuple):
            endpoint = Endpoint.loopback(1)  # the port is closed
        started = time.monotonic()
        with pytest.raises(IpcConnectError):
            connect_channel(endpoint, key, "echo", connect_timeout_s=0.3)
        assert 0.15 <= time.monotonic() - started < 20.0

    def test_the_client_retries_until_the_server_comes_up(
        self, native: Endpoint, key: ConnectionKey, servers: list[IpcServer]
    ) -> None:
        server = IpcServer(native, key, {"echo": EchoChannel()})
        timer = threading.Timer(0.4, server.start)
        timer.start()
        servers.append(server)
        try:
            wire, _ = connect_channel(native, key, "echo", connect_timeout_s=10.0)
            wire.close()
        finally:
            timer.join()


class TestHostilePeers:
    """Raw connections that skip or abuse the protocol."""

    def raw(self, server: IpcServer) -> Any:
        return Client(server.endpoint.address, family=server.endpoint.family)

    def test_a_silent_peer_is_dropped_after_the_handshake_timeout(
        self, start_server: Callable[..., IpcServer]
    ) -> None:
        server = start_server({"echo": EchoChannel()}, handshake_timeout_s=0.3)
        peer = self.raw(server)
        try:
            assert peer.poll(5.0)  # the challenge
            peer.recv_bytes(256)
            assert wait_until(lambda: server.stats.auth_failures == 1)
            with pytest.raises((EOFError, OSError)):
                peer.recv_bytes(256)  # the server closed the connection
        finally:
            peer.close()

    def test_a_peer_that_sends_a_pickle_is_rejected_and_nothing_runs(
        self, start_server: Callable[..., IpcServer]
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        peer = self.raw(server)
        try:
            peer.recv_bytes(256)
            peer.send_bytes(pickle.dumps(Evil()))
            assert wait_until(lambda: server.stats.auth_failures == 1)
        finally:
            peer.close()
        assert PICKLE_RAN == []

    def test_an_oversized_first_message_closes_the_connection(
        self, start_server: Callable[..., IpcServer]
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        peer = self.raw(server)
        try:
            peer.recv_bytes(256)
            peer.send_bytes(b"x" * 4096)  # the handshake accepts 128 bytes
            assert wait_until(lambda: server.stats.auth_failures == 1)
        finally:
            peer.close()

    def test_a_recorded_answer_does_not_work_on_a_new_connection(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        first = self.raw(server)
        try:
            challenge = first.recv_bytes(256)
            nonce = challenge[len(MAGIC) + 1 :]
            client_nonce = b"c" * 32
            answer = MAGIC + b"\x02" + key.sign(b"client", nonce, client_nonce) + client_nonce
            first.send_bytes(answer)
            welcome = first.recv_bytes(256)  # the real answer is accepted
            assert welcome.startswith(MAGIC + b"\x03")
        finally:
            first.close()
        second = self.raw(server)
        try:
            second.recv_bytes(256)
            second.send_bytes(answer)  # replayed
            assert second.recv_bytes(256) == MAGIC + b"\x00"
        finally:
            second.close()
        assert wait_until(lambda: server.stats.auth_failures == 1)

    def test_a_reflected_challenge_is_not_an_answer(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        peer = self.raw(server)
        try:
            challenge = peer.recv_bytes(256)
            nonce = challenge[len(MAGIC) + 1 :]
            # Answer with the proof that the server itself would give, labeled as the server.
            forged = MAGIC + b"\x02" + key.sign(b"server", nonce, nonce) + nonce
            peer.send_bytes(forged)
            assert peer.recv_bytes(256) == MAGIC + b"\x00"
        finally:
            peer.close()

    def test_waiting_handshakes_are_capped(self, start_server: Callable[..., IpcServer]) -> None:
        server = start_server({"echo": EchoChannel()}, handshake_timeout_s=1.0, max_pending=2)
        peers = [self.raw(server) for _ in range(5)]
        try:
            assert wait_until(lambda: server.stats.refused_busy >= 1)
        finally:
            for peer in peers:
                peer.close()

    def test_a_bad_hello_is_refused_and_counted(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        wire = self._authenticated(server, key)
        with wire:
            wire.send(b"not json")
            assert wait_until(lambda: server.stats.hello_failures == 1)

    def test_a_hello_with_the_wrong_version_is_refused(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        server = start_server({"echo": EchoChannel()})
        wire = self._authenticated(server, key)
        with wire:
            wire.send(encode_json({"v": 99, "channel": "echo", "params": {}}))
            reply = decode_json(wire.recv(5.0) or b"")
            assert reply["ok"] is False

    def _authenticated(self, server: IpcServer, key: ConnectionKey) -> Wire:
        from seeingmon.services.ipc.handshake import client_handshake

        connection = Client(server.endpoint.address, family=server.endpoint.family)
        wire = Wire(connection, max_message_bytes=1 << 20)
        client_handshake(wire, key, timeout_s=5.0)
        return wire


class TestServerLifecycle:
    def test_stop_releases_the_address_for_a_new_server(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        first = IpcServer(native, key, {"echo": EchoChannel()})
        first.start()
        wire, _ = connect_channel(native, key, "echo")
        wire.close()
        first.stop()
        assert not first.running
        second = IpcServer(native, key, {"echo": EchoChannel()})

        def started() -> bool:
            # On Windows, the server end of the connection that the client closed can outlive
            # `stop` by a few milliseconds, and the pipe name stays taken until it is gone.
            try:
                second.start()
            except IpcAddressInUseError:
                return False
            return True

        assert wait_until(started, timeout_s=10.0, interval_s=0.05)
        try:
            wire, _ = connect_channel(native, key, "echo")
            wire.close()
        finally:
            second.stop()

    def test_stop_is_safe_to_call_twice_and_before_start(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        server = IpcServer(native, key, {})
        server.stop()
        server.start()
        server.stop()
        server.stop()

    def test_a_server_cannot_start_twice(self, native: Endpoint, key: ConnectionKey) -> None:
        server = IpcServer(native, key, {})
        server.start()
        try:
            with pytest.raises(RuntimeError, match="already"):
                server.start()
        finally:
            server.stop()

    def test_two_servers_cannot_share_an_address(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        first = IpcServer(native, key, {})
        first.start()
        try:
            with pytest.raises(IpcAddressInUseError):
                IpcServer(native, key, {}).start()
        finally:
            first.stop()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets")
class TestUnixSocketFile:
    def test_the_socket_file_is_private_and_removed_on_stop(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        assert native.path is not None
        server = IpcServer(native, key, {})
        server.start()
        mode = stat.S_IMODE(os.stat(native.path).st_mode)
        assert mode & 0o077 == 0
        server.stop()
        assert not os.path.exists(native.path)

    def test_a_stale_socket_file_from_a_killed_process_is_replaced(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        assert native.path is not None
        stale = socket.socket(socket.AddressFamily["AF_UNIX"])
        stale.bind(native.path)
        stale.close()  # the file stays, as it does after a SIGKILL
        assert os.path.exists(native.path)
        server = IpcServer(native, key, {"echo": EchoChannel()})
        server.start()
        try:
            wire, _ = connect_channel(native, key, "echo")
            wire.close()
        finally:
            server.stop()

    def test_a_file_that_is_not_a_socket_is_never_removed(
        self, native: Endpoint, key: ConnectionKey
    ) -> None:
        assert native.path is not None
        Path(native.path).write_text("precious")
        with pytest.raises(IpcAddressInUseError, match="not a socket"):
            IpcServer(native, key, {}).start()
        assert Path(native.path).read_text() == "precious"

    def test_the_directory_is_created_private(self, short_dir: Path, key: ConnectionKey) -> None:
        endpoint = Endpoint(str(short_dir / "run" / "a.sock"), FAMILY_UNIX)
        server = IpcServer(endpoint, key, {})
        server.start()
        try:
            mode = stat.S_IMODE(os.stat(short_dir / "run").st_mode)
            assert mode & 0o077 == 0
        finally:
            server.stop()


class TestWire:
    @pytest.fixture
    def pair(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> Iterator[tuple[Wire, EchoChannel]]:
        channel = EchoChannel(max_message_bytes=32 << 20)
        server = start_server({"echo": channel})
        wire, _ = connect_channel(server.endpoint, key, "echo", max_message_bytes=32 << 20)
        yield wire, channel
        wire.close()

    def test_a_large_message_survives(self, pair: tuple[Wire, EchoChannel]) -> None:
        wire, _ = pair
        payload = os.urandom(5 * 1024 * 1024)
        wire.send(payload)
        assert wire.recv(20.0) == payload

    def test_recv_returns_none_when_nothing_arrives(self, pair: tuple[Wire, EchoChannel]) -> None:
        wire, _ = pair
        started = time.monotonic()
        assert wire.recv(0.25) is None
        assert 0.15 <= time.monotonic() - started < 20.0
        assert wire.recv(0) is None

    def test_a_message_over_the_limit_is_refused_when_sending(
        self, pair: tuple[Wire, EchoChannel]
    ) -> None:
        wire, _ = pair
        wire.max_message_bytes = 100
        with pytest.raises(IpcProtocolError):
            wire.send(b"x" * 101)
        wire.send(b"x" * 100)

    def test_a_message_over_the_limit_closes_the_receiving_wire(
        self, start_server: Callable[..., IpcServer], key: ConnectionKey
    ) -> None:
        channel = EchoChannel(max_message_bytes=1000)
        server = start_server({"echo": channel})
        wire, _ = connect_channel(server.endpoint, key, "echo", max_message_bytes=32 << 20)
        with wire:
            wire.send(b"y" * 5000)
            assert wait_until(lambda: len(channel.wires) == 1 and channel.wires[0].closed)
            with pytest.raises((IpcClosedError, IpcProtocolError)):
                drain(wire)

    def test_close_wakes_a_blocked_receive_quickly(self, pair: tuple[Wire, EchoChannel]) -> None:
        wire, _ = pair
        outcome: list[BaseException | None] = []

        def receive() -> None:
            try:
                wire.recv(None)
                outcome.append(None)
            except IpcClosedError as error:
                outcome.append(error)

        thread = threading.Thread(target=receive)
        thread.start()
        time.sleep(0.2)
        started = time.monotonic()
        wire.close()
        thread.join(5.0)
        assert not thread.is_alive()
        assert isinstance(outcome[0], IpcClosedError)
        assert time.monotonic() - started < 4.0

    def test_the_peer_closing_is_a_closed_error(self, pair: tuple[Wire, EchoChannel]) -> None:
        wire, channel = pair
        assert wait_until(lambda: len(channel.wires) == 1)
        channel.wires[0].close()
        with pytest.raises(IpcClosedError):
            drain(wire)
        assert wire.closed

    def test_sending_on_a_closed_wire_raises(self, pair: tuple[Wire, EchoChannel]) -> None:
        wire, _ = pair
        wire.close("test over")
        with pytest.raises(IpcClosedError, match="test over"):
            wire.send(b"x")
        with pytest.raises(IpcClosedError):
            wire.recv(0.1)
        wire.close()  # twice is fine

    def test_concurrent_senders_keep_messages_whole(self, pair: tuple[Wire, EchoChannel]) -> None:
        wire, _ = pair
        count = 40

        def send(tag: bytes) -> None:
            for _ in range(count):
                wire.send(tag * 20_000)

        threads = [threading.Thread(target=send, args=(bytes([65 + i]),)) for i in range(4)]
        for thread in threads:
            thread.start()
        received = []
        for _ in range(count * 4):
            message = wire.recv(20.0)
            assert message is not None
            received.append(message)
        for thread in threads:
            thread.join()
        assert all(len(set(message)) == 1 and len(message) == 20_000 for message in received)

    def test_counters_follow_the_traffic(self, pair: tuple[Wire, EchoChannel]) -> None:
        wire, _ = pair
        sent_before, bytes_before = wire.messages_sent, wire.bytes_sent
        received_before = wire.messages_received
        wire.send(b"12345")
        assert wire.recv(5.0) == b"12345"
        assert wire.messages_sent - sent_before == 1
        assert wire.bytes_sent - bytes_before == 5
        assert wire.messages_received - received_before == 1
