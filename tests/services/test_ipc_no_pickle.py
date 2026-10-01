"""No pickle crosses a process boundary: a runtime guard, a hostile peer, and a source scan."""

from __future__ import annotations

import ast
import multiprocessing
import pickle
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import seeingmon.services
from seeingmon.drivers.base import CameraStateError
from seeingmon.frames import Frame, Roi, TimeQuality, decode_frame, encode_frame, frames_equal
from seeingmon.services.ipc import (
    IpcServer,
    RpcService,
    StreamSender,
    StreamService,
    StreamWindow,
    connect_channel,
    connect_rpc,
    connect_stream,
)
from seeingmon.services.ipc.errors import IpcClosedError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.wire import Wire

from . import no_pickle

RAN: list[int] = []


def _run_me() -> None:
    RAN.append(1)


class Evil:
    """Unpickling an instance calls `_run_me`, so `RAN` shows whether anything unpickled."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (_run_me, ())


EVIL_BYTES = pickle.dumps(Evil())  # built before any guard is installed


@pytest.fixture
def guard() -> Iterator[list[str]]:
    """Forbid pickle for the duration of the test, and give back the record of calls."""
    no_pickle.CALLS.clear()
    restore = no_pickle.install()
    try:
        yield no_pickle.CALLS
    finally:
        restore()


def a_frame(seq: int) -> Frame:
    return Frame(
        data=np.full((2, 8), seq, dtype=np.uint16),
        stream_id=3,
        seq=seq,
        t_arrival_ns=1_000 + seq,
        t_utc_ns=900 + seq,
        t_err_ns=50,
        t_quality=TimeQuality.FITTED,
        dropped_before=0,
        exposure_us=1000,
        gain=100,
        mode="bin1",
        roi=Roi(0, 0, 8, 2),
        adc_bits=14,
    )


def test_the_guard_catches_every_way_to_pickle(guard: list[str]) -> None:
    first, second = multiprocessing.Pipe()
    try:
        with pytest.raises(AssertionError, match="pickle"):
            first.send({"a": 1})
        with pytest.raises(AssertionError, match="pickle"):
            pickle.dumps({"a": 1})
        with pytest.raises(AssertionError, match="pickle"):
            pickle.loads(EVIL_BYTES)
    finally:
        first.close()
        second.close()
    assert any("send" in call for call in guard)
    assert any("dumps" in call for call in guard)
    assert any("loads" in call for call in guard)


def test_rpc_and_streams_run_without_any_pickle(
    guard: list[str], start_server: Callable[..., IpcServer], key: ConnectionKey
) -> None:
    senders: list[StreamSender] = []
    arrived = threading.Event()

    def on_sender(sender: StreamSender, params: Mapping[str, Any]) -> None:
        senders.append(sender)
        arrived.set()

    def fail(params: Mapping[str, Any]) -> Any:
        raise CameraStateError("not capturing")

    rpc = RpcService({"echo": lambda params: dict(params), "fail": fail})
    rpc.start()
    server = start_server({"rpc": rpc, "frames": StreamService(on_sender)})
    client, _ = connect_rpc(server.endpoint, key)
    receiver, _ = connect_stream(
        server.endpoint, key, channel="frames", window=StreamWindow(8, 1 << 20)
    )
    try:
        assert client.call("echo", {"a": [1, 2, {"b": None}]}) == {"a": [1, 2, {"b": None}]}
        with pytest.raises(CameraStateError):
            client.call("fail")
        assert arrived.wait(10.0)
        for seq in range(5):
            senders[0].send(encode_frame(a_frame(seq)), tag=1)
        for seq in range(5):
            message = receiver.recv(10.0)
            assert message is not None
            assert frames_equal(decode_frame(message.payload), a_frame(seq))
    finally:
        receiver.close()
        client.close()
        rpc.stop()
    assert guard == []


def test_a_pickle_from_an_authenticated_peer_is_never_loaded(
    start_server: Callable[..., IpcServer], key: ConnectionKey
) -> None:
    rpc = RpcService({"echo": lambda params: dict(params)})
    rpc.start()
    server = start_server({"rpc": rpc})
    wire, _ = connect_channel(server.endpoint, key, "rpc")
    try:
        with wire:
            wire.send(EVIL_BYTES)
            assert wire.recv(10.0) is not None  # the service answers with a JSON error
    finally:
        rpc.stop()
    assert RAN == []


def test_a_pickle_on_the_stream_channel_is_never_loaded(
    start_server: Callable[..., IpcServer], key: ConnectionKey
) -> None:
    senders: list[StreamSender] = []
    arrived = threading.Event()

    def on_sender(sender: StreamSender, params: Mapping[str, Any]) -> None:
        senders.append(sender)
        arrived.set()

    server = start_server({"frames": StreamService(on_sender)})
    receiver, _ = connect_stream(server.endpoint, key, channel="frames")
    try:
        assert arrived.wait(10.0)
        malicious: Wire = senders[0]._wire  # a sender that skips the stream protocol
        malicious.send(EVIL_BYTES)
        with pytest.raises(IpcClosedError):
            receiver.recv(10.0)  # the receiver closes on the malformed message
    finally:
        receiver.close()
    assert RAN == []


FORBIDDEN_MODULES = {"pickle", "_pickle", "cPickle", "cloudpickle", "dill", "marshal", "shelve"}
ALLOWED_MULTIPROCESSING = {"Client", "Listener"}

# The survey worker is a process pool, and the pool pickles what it sends. Only plain data crosses
# that boundary (the pipeline specification, the encoded frame, and the dictionaries of the result,
# see `seeingmon.survey.analyzer`). No connection between `acquire`, `core`, and `web` uses it, and
# the module still may not import a pickle module.
POOL_USERS = {"survey_worker.py"}


def _sources() -> list[Path]:
    root = Path(seeingmon.services.__file__).parent
    return sorted(root.rglob("*.py"))


@pytest.mark.parametrize("path", _sources(), ids=lambda path: path.name)
def test_the_source_imports_nothing_that_pickles(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in FORBIDDEN_MODULES, f"{path.name} imports {alias.name}"
                assert (
                    alias.name in ("multiprocessing.connection",)
                    or root != "multiprocessing"
                    or path.name in POOL_USERS
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in FORBIDDEN_MODULES, f"{path.name} imports from {node.module}"
            if root == "multiprocessing" and path.name not in POOL_USERS:
                assert node.module == "multiprocessing.connection", path.name
                assert {alias.name for alias in node.names} <= ALLOWED_MULTIPROCESSING, path.name
