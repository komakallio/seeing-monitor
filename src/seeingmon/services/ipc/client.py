"""The connecting side of the connection layer.

`connect_channel` connects to an `IpcServer`, proves that it holds the connection key, and
asks for a channel. It returns the wire, ready for the channel's own messages, and the `reply`
that the channel handler sent back. `connect_rpc` and `connect_stream` build on it.

While the server is not up yet (a process that restarts), the connection attempt retries until
`connect_timeout_s` passes, and then raises `IpcConnectError`.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from multiprocessing.connection import Client
from typing import Any

from seeingmon.clock import NS_PER_S, Clock, SystemClock
from seeingmon.services.ipc.codec import as_mapping, decode_json, encode_json
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import IpcAuthError, IpcConnectError, IpcError, IpcProtocolError
from seeingmon.services.ipc.handshake import client_handshake
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import DEFAULT_MESSAGE_BYTES, HELLO_VERSION, MAX_HELLO_BYTES
from seeingmon.services.ipc.wire import Wire

RETRY_INTERVAL_S = 0.05
_REAL_CLOCK = SystemClock()


def connect_channel(
    endpoint: Endpoint,
    key: ConnectionKey,
    channel: str,
    params: Mapping[str, Any] | None = None,
    *,
    connect_timeout_s: float = 5.0,
    handshake_timeout_s: float = 5.0,
    max_message_bytes: int = DEFAULT_MESSAGE_BYTES,
    clock: Clock | None = None,
) -> tuple[Wire, Mapping[str, Any]]:
    """Connect, authenticate, and open `channel`. Returns the wire and the handler's reply.

    Raises `IpcConnectError` when nobody listens within `connect_timeout_s`, `IpcAuthError`
    when the keys differ, and `IpcProtocolError` when the server refuses the channel.
    """
    clock = _REAL_CLOCK if clock is None else clock
    deadline_ns = clock.monotonic_ns() + round(connect_timeout_s * NS_PER_S)
    waiter = threading.Event()
    while True:
        try:
            connection = Client(endpoint.address, family=endpoint.family)
            break
        except OSError as error:
            if clock.monotonic_ns() >= deadline_ns:
                raise IpcConnectError(f"nothing answers at {endpoint}") from error
            waiter.wait(RETRY_INTERVAL_S)
    wire = Wire(connection, max_message_bytes=max_message_bytes, name=channel, clock=clock)
    try:
        client_handshake(wire, key, timeout_s=handshake_timeout_s)
        wire.send(
            encode_json({"v": HELLO_VERSION, "channel": channel, "params": dict(params or {})})
        )
        raw = wire.recv(handshake_timeout_s, max_bytes=MAX_HELLO_BYTES)
        if raw is None:
            raise IpcProtocolError("the server did not answer the hello in time")
        data = as_mapping(decode_json(raw), "hello reply")
        if data.get("ok") is not True:
            error_text = "no reason given"
            detail = data.get("error")
            if isinstance(detail, Mapping) and isinstance(detail.get("message"), str):
                error_text = detail["message"]
            raise IpcProtocolError(f"the server refused the {channel} channel: {error_text}")
        reply = as_mapping(data.get("reply", {}), "hello reply.reply")
    except IpcAuthError:
        wire.close()
        raise
    except IpcError as error:
        wire.close()
        if isinstance(error, IpcProtocolError):
            raise
        raise IpcConnectError(f"the connection to {endpoint} failed: {error}") from error
    except Exception:
        wire.close()
        raise
    return wire, reply
