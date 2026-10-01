"""The request-response layer."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from seeingmon.drivers.base import (
    CameraStateError,
    CameraTimeoutError,
)
from seeingmon.frames import Roi
from seeingmon.services.ipc.client import connect_channel
from seeingmon.services.ipc.codec import decode_json, decode_roi, encode_json, encode_roi
from seeingmon.services.ipc.errors import (
    IpcClosedError,
    IpcProtocolError,
    RemoteError,
    RpcInvalidParamsError,
    RpcMethodNotFoundError,
    RpcTimeoutError,
)
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import (
    Handler,
    RpcClient,
    RpcConnection,
    RpcService,
    connect_rpc,
)
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.ipc.wire import Wire

from .conftest import drain, wait_until


class Harness:
    """A service with a few handlers, behind a server, and the clients that connected."""

    def __init__(
        self,
        start_server: Callable[..., IpcServer],
        key: ConnectionKey,
        handlers: Mapping[str, Handler],
        **options: Any,
    ) -> None:
        self.key = key
        self.service = RpcService(handlers, **options)
        self.service.start()
        self.server = start_server({"rpc": self.service})
        self.clients: list[RpcClient] = []

    def client(self, **options: Any) -> RpcClient:
        client, _ = connect_rpc(self.server.endpoint, self.key, **options)
        self.clients.append(client)
        return client

    def close(self) -> None:
        for client in self.clients:
            client.close()
        self.service.stop()


def make_handlers(release: threading.Event, log: list[str]) -> dict[str, Handler]:
    def add(params: Mapping[str, Any]) -> Any:
        return params["a"] + params["b"]

    def block(params: Mapping[str, Any]) -> Any:
        log.append("block:start")
        release.wait(30.0)
        log.append("block:end")
        return "released"

    def record(params: Mapping[str, Any]) -> Any:
        log.append(str(params["n"]))
        return params["n"]

    def camera_timeout(params: Mapping[str, Any]) -> Any:
        raise CameraTimeoutError("no frame in 2 s")

    def camera_state(params: Mapping[str, Any]) -> Any:
        raise CameraStateError("configure before open")

    def unexpected(params: Mapping[str, Any]) -> Any:
        raise ZeroDivisionError("division by zero")

    def not_json(params: Mapping[str, Any]) -> Any:
        return {"value": object()}

    def roi(params: Mapping[str, Any]) -> Any:
        decoded = decode_roi(params["roi"])  # a malformed ROI raises CodecError
        return encode_roi(Roi(decoded.x + 1, decoded.y, decoded.width, decoded.height))

    def big(params: Mapping[str, Any]) -> Any:
        return "x" * (2 * 1024 * 1024)

    return {
        "add": add,
        "block": block,
        "record": record,
        "camera_timeout": camera_timeout,
        "camera_state": camera_state,
        "unexpected": unexpected,
        "not_json": not_json,
        "roi": roi,
        "big": big,
        "ping": lambda params: "pong",
        "none": lambda params: None,
    }


@pytest.fixture
def release() -> Iterator[threading.Event]:
    event = threading.Event()
    yield event
    event.set()  # never leave a worker blocked


@pytest.fixture
def log() -> list[str]:
    return []


@pytest.fixture
def harness(
    start_server: Callable[..., IpcServer],
    key: ConnectionKey,
    release: threading.Event,
    log: list[str],
) -> Iterator[Harness]:
    instance = Harness(start_server, key, make_handlers(release, log), inline={"ping"})
    yield instance
    instance.close()


class TestCalls:
    def test_a_call_returns_the_result(self, harness: Harness) -> None:
        client = harness.client()
        assert client.call("add", {"a": 2, "b": 3}) == 5
        assert client.call("none") is None
        assert client.call("ping") == "pong"

    def test_the_hello_reply_names_the_connection(
        self, harness: Harness, key: ConnectionKey
    ) -> None:
        client, reply = connect_rpc(harness.server.endpoint, key)
        harness.clients.append(client)
        assert isinstance(reply["connection"], int)

    def test_an_unknown_method_raises_method_not_found(self, harness: Harness) -> None:
        with pytest.raises(RpcMethodNotFoundError, match="no method named 'nope'"):
            harness.client().call("nope")

    def test_registered_exceptions_come_back_as_their_own_class(self, harness: Harness) -> None:
        client = harness.client()
        with pytest.raises(CameraTimeoutError, match="no frame in 2 s"):
            client.call("camera_timeout")
        with pytest.raises(CameraStateError, match="configure before open"):
            client.call("camera_state")
        assert client.call("add", {"a": 1, "b": 1}) == 2  # the connection survives

    def test_an_unregistered_exception_arrives_as_a_remote_error(self, harness: Harness) -> None:
        with pytest.raises(RemoteError, match="ZeroDivisionError: division by zero") as raised:
            harness.client().call("unexpected")
        assert raised.value.remote_type == "InternalError"

    def test_a_result_that_is_not_json_is_an_error_and_not_a_hang(self, harness: Harness) -> None:
        client = harness.client()
        with pytest.raises(TypeError, match="not JSON serializable"):
            client.call("not_json", timeout_s=10.0)
        assert client.call("add", {"a": 1, "b": 2}) == 3

    def test_bad_parameters_raise_invalid_params(self, harness: Harness) -> None:
        client = harness.client()
        with pytest.raises(RpcInvalidParamsError, match="roi"):
            client.call("roi", {"roi": {"x": "one"}})
        assert client.call("roi", {"roi": encode_roi(Roi(1, 2, 3, 4))}) == encode_roi(
            Roi(2, 2, 3, 4)
        )

    def test_a_result_over_the_limit_is_an_error_and_not_a_hang(self, harness: Harness) -> None:
        client = harness.client()
        with pytest.raises(RemoteError, match="exceeds the message limit"):
            client.call("big", timeout_s=10.0)

    def test_params_must_serialize(self, harness: Harness) -> None:
        client = harness.client()
        with pytest.raises(TypeError):
            client.call("add", {"a": object(), "b": 1})
        assert client.call("add", {"a": 1, "b": 1}) == 2

    def test_a_timeout_must_be_positive(self, harness: Harness) -> None:
        with pytest.raises(ValueError, match="positive"):
            harness.client().call("ping", timeout_s=0)


class TestTimeoutsAndConcurrency:
    def test_a_timed_out_call_leaves_the_connection_usable(
        self, harness: Harness, release: threading.Event, log: list[str]
    ) -> None:
        client = harness.client()
        started = time.monotonic()
        with pytest.raises(RpcTimeoutError, match="block did not answer"):
            client.call("block", timeout_s=0.3)
        assert 0.15 <= time.monotonic() - started < 20.0
        assert client.call("ping") == "pong"  # inline methods answer while a worker is busy
        release.set()
        assert wait_until(lambda: "block:end" in log)  # the late answer is dropped, not misrouted
        assert client.call("add", {"a": 4, "b": 5}) == 9

    def test_many_threads_call_at_once_and_each_gets_its_own_answer(self, harness: Harness) -> None:
        client = harness.client()
        failures: list[str] = []

        def work(seed: int) -> None:
            for index in range(60):
                result = client.call("add", {"a": seed * 1000, "b": index})
                if result != seed * 1000 + index:
                    failures.append(f"{seed}:{index}:{result}")

        threads = [threading.Thread(target=work, args=(seed,)) for seed in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60.0)
        assert not any(thread.is_alive() for thread in threads)
        assert failures == []

    def test_one_worker_runs_requests_in_the_order_they_arrive(
        self, harness: Harness, log: list[str]
    ) -> None:
        client = harness.client()
        for number in range(20):
            client.call("record", {"n": number})
        assert log == [str(number) for number in range(20)]

    def test_workers_run_handlers_in_parallel(
        self,
        start_server: Callable[..., IpcServer],
        key: ConnectionKey,
        release: threading.Event,
        log: list[str],
    ) -> None:
        barrier = threading.Barrier(3, timeout=10.0)
        results: list[Any] = []

        def meet(params: Mapping[str, Any]) -> Any:
            return barrier.wait()

        harness = Harness(start_server, key, {"meet": meet}, workers=3)
        try:
            client = harness.client()

            def call() -> None:
                results.append(client.call("meet", timeout_s=15.0))

            threads = [threading.Thread(target=call) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(20.0)
            assert sorted(results) == [0, 1, 2]
        finally:
            harness.close()

    def test_submitted_jobs_run_on_a_worker_after_the_queued_requests(
        self, harness: Harness, log: list[str]
    ) -> None:
        client = harness.client()
        client.call("record", {"n": 1})
        harness.service.submit(lambda: log.append("job"))
        client.call("record", {"n": 2})
        assert log == ["1", "job", "2"]

    def test_the_busy_service_refuses_requests_beyond_its_queue(
        self, harness: Harness, key: ConnectionKey, release: threading.Event
    ) -> None:
        wire, _ = connect_channel(harness.server.endpoint, key, "rpc")
        refused: list[int] = []

        def read_answers() -> None:
            try:
                while True:
                    raw = wire.recv(0.2)
                    if raw is not None and decode_json(raw).get("ok") is False:
                        refused.append(1)
            except IpcClosedError:
                pass

        reader = threading.Thread(target=read_answers)
        reader.start()
        try:
            wire.send(encode_json({"v": 1, "id": 1, "method": "block", "params": None}))
            for number in range(2, 400):
                wire.send(encode_json({"v": 1, "id": number, "method": "none", "params": None}))
            assert wait_until(lambda: len(refused) >= 1, 15.0)
        finally:
            release.set()
            wire.close()
            reader.join(10.0)
        assert not reader.is_alive()


class TestPeerDisappears:
    def test_a_pending_call_fails_when_the_service_closes_the_connection(
        self, harness: Harness, release: threading.Event, log: list[str]
    ) -> None:
        client = harness.client()
        outcome: list[BaseException] = []

        def call() -> None:
            try:
                client.call("block", timeout_s=20.0)
            except BaseException as error:
                outcome.append(error)

        thread = threading.Thread(target=call)
        thread.start()
        assert wait_until(lambda: "block:start" in log)  # the call is in flight
        harness.service.connections[0].close("restarting")
        thread.join(10.0)
        assert not thread.is_alive()
        assert isinstance(outcome[0], IpcClosedError)
        assert wait_until(lambda: client.closed)
        with pytest.raises(IpcClosedError):
            client.call("ping")

    def test_stopping_the_server_side_fails_calls_quickly(
        self, harness: Harness, release: threading.Event
    ) -> None:
        client = harness.client()
        assert client.call("ping") == "pong"
        release.set()
        harness.service.stop()
        with pytest.raises(IpcClosedError):
            client.call("ping", timeout_s=10.0)

    def test_closing_the_client_tells_the_service_once(
        self,
        start_server: Callable[..., IpcServer],
        key: ConnectionKey,
    ) -> None:
        gone: list[RpcConnection] = []
        harness = Harness(start_server, key, {"ping": lambda params: 1}, on_disconnect=gone.append)
        try:
            client = harness.client()
            assert client.call("ping") == 1
            client.close()
            assert wait_until(lambda: len(gone) == 1)
            client.close()
            time.sleep(0.3)
            assert len(gone) == 1
            assert harness.service.connections == []
        finally:
            harness.close()

    def test_a_dead_service_raises_closed_error_on_the_next_call(
        self, harness: Harness, release: threading.Event
    ) -> None:
        client = harness.client()
        harness.server.stop()
        harness.service.stop()
        with pytest.raises(IpcClosedError):
            client.call("ping", timeout_s=10.0)


class TestConnections:
    def test_a_new_client_replaces_the_old_one_when_the_limit_is_one(
        self,
        start_server: Callable[..., IpcServer],
        key: ConnectionKey,
    ) -> None:
        gone: list[int] = []
        harness = Harness(
            start_server,
            key,
            {"ping": lambda params: "pong"},
            max_connections=1,
            on_disconnect=lambda connection: gone.append(connection.number),
        )
        try:
            first = harness.client()
            assert first.call("ping") == "pong"
            second = harness.client()
            assert second.call("ping") == "pong"
            assert wait_until(lambda: first.closed)
            with pytest.raises(IpcClosedError):
                first.call("ping")
            assert wait_until(lambda: len(gone) == 1)
        finally:
            harness.close()

    def test_two_clients_can_connect_when_the_limit_allows(
        self,
        start_server: Callable[..., IpcServer],
        key: ConnectionKey,
    ) -> None:
        harness = Harness(start_server, key, {"ping": lambda params: "pong"}, max_connections=2)
        try:
            first, second = harness.client(), harness.client()
            assert first.call("ping") == second.call("ping") == "pong"
        finally:
            harness.close()

    def test_on_connect_adds_to_the_reply_and_can_refuse(
        self,
        start_server: Callable[..., IpcServer],
        key: ConnectionKey,
    ) -> None:
        def on_connect(connection: RpcConnection) -> Mapping[str, Any] | None:
            if connection.params.get("refuse"):
                raise IpcProtocolError("not today")
            connection.context["who"] = connection.params.get("who")
            return {"session": "s-1"}

        harness = Harness(start_server, key, {"ping": lambda params: 1}, on_connect=on_connect)
        try:
            client, reply = connect_rpc(harness.server.endpoint, key, {"who": "tester"})
            harness.clients.append(client)
            assert reply["session"] == "s-1"
            assert harness.service.connections[0].context == {"who": "tester"}
            with pytest.raises(IpcProtocolError, match="not today"):
                connect_rpc(harness.server.endpoint, key, {"refuse": True})
            assert client.call("ping") == 1  # the refused client did not disturb the first
        finally:
            harness.close()


class TestMalformedRequests:
    def raw(self, harness: Harness, key: ConnectionKey) -> Wire:
        wire, _ = connect_channel(harness.server.endpoint, key, "rpc")
        return wire

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param(b"not json", id="not-json"),
            pytest.param(encode_json({"v": 1, "id": 1, "params": {}}), id="no-method"),
            pytest.param(encode_json({"v": 2, "id": 1, "method": "ping"}), id="wrong-version"),
            pytest.param(encode_json({"v": 1, "id": "1", "method": "ping"}), id="string-id"),
            pytest.param(
                encode_json({"v": 1, "id": 1, "method": "ping", "params": [1]}), id="list"
            ),
            pytest.param(encode_json({"v": 1, "id": 1, "method": "p" * 500}), id="long-method"),
            pytest.param(
                b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00\x8c\x01a\x94.", id="pickle"
            ),
        ],
    )
    def test_a_request_that_breaks_the_protocol_ends_the_session(
        self, harness: Harness, key: ConnectionKey, message: bytes
    ) -> None:
        wire = self.raw(harness, key)
        with wire:
            wire.send(message)
            answer = decode_json(wire.recv(5.0) or b"")
            assert answer["ok"] is False
            assert answer["id"] is None
            with pytest.raises(IpcClosedError):
                drain(wire)
        assert wait_until(lambda: harness.service.connections == [])

    def test_an_oversized_request_ends_the_session(
        self,
        start_server: Callable[..., IpcServer],
        key: ConnectionKey,
    ) -> None:
        harness = Harness(start_server, key, {"ping": lambda params: 1}, max_message_bytes=2048)
        try:
            wire = self.raw(harness, key)
            with wire:
                wire.max_message_bytes = 1 << 20
                wire.send(b"{" + b" " * 4096 + b"}")
                with pytest.raises(IpcClosedError):
                    drain(wire)
            assert wait_until(lambda: harness.service.connections == [])
        finally:
            harness.close()
