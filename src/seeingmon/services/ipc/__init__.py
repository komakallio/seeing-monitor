"""The connection layer: authenticated local connections, RPC, and frame streams.

The layer is built on `multiprocessing.connection`, and only on `send_bytes`, `recv_bytes`, and
`poll`. Nothing here calls `send` or `recv`, which pickle, and nothing here imports `pickle`. A
test fails if any pickle runs.

**Server side.** One `IpcServer` listens at one `Endpoint` and serves named channels. A client
proves that it holds the connection key, names a channel, and gets its own connection for it,
so different channels share nothing.

    server = IpcServer(endpoint, key, {"rpc": rpc_service, "frames": stream_service})
    server.start()

- `RpcService` serves requests and answers (`seeingmon.services.ipc.rpc`).
- `StreamService` gives each stream client to the application as a `StreamSender`
  (`seeingmon.services.ipc.stream`).

**Client side.** `connect_rpc` returns an `RpcClient`, `connect_stream` returns a
`StreamReceiver`, and `connect_channel` is the building block for a new kind of channel.

    client, hello = connect_rpc(endpoint, key)
    result = client.call("method", {"parameter": 1})

**Messages.** Requests, answers, and errors are JSON. The codecs for the camera types and
exceptions are in `seeingmon.services.ipc.codec`. A stream carries opaque payloads, for example
`seeingmon.frames.encode_frame` bytes, with consecutive sequence numbers and a flow-control
window.
"""

from __future__ import annotations

from seeingmon.services.ipc.client import connect_channel
from seeingmon.services.ipc.codec import (
    DEFAULT_ERRORS,
    CodecError,
    ErrorRegistry,
    decode_active_stream,
    decode_camera_caps,
    decode_camera_info,
    decode_exception,
    decode_json,
    decode_recovery_level,
    decode_roi,
    decode_stream_config,
    encode_active_stream,
    encode_camera_caps,
    encode_camera_info,
    encode_exception,
    encode_json,
    encode_roi,
    encode_stream_config,
)
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import (
    IpcAddressInUseError,
    IpcAuthError,
    IpcClosedError,
    IpcConfigError,
    IpcConnectError,
    IpcError,
    IpcProtocolError,
    RemoteError,
    RpcError,
    RpcInvalidParamsError,
    RpcMethodNotFoundError,
    RpcTimeoutError,
    StreamCreditError,
    StreamError,
)
from seeingmon.services.ipc.keys import ConnectionKey, load_connection_key
from seeingmon.services.ipc.rpc import RpcClient, RpcConnection, RpcService, connect_rpc
from seeingmon.services.ipc.server import Accepted, ChannelHandler, IpcServer
from seeingmon.services.ipc.stream import (
    StreamKind,
    StreamMessage,
    StreamReceiver,
    StreamSender,
    StreamService,
    StreamWindow,
    connect_stream,
)
from seeingmon.services.ipc.wire import Wire

__all__ = [
    "DEFAULT_ERRORS",
    "Accepted",
    "ChannelHandler",
    "CodecError",
    "ConnectionKey",
    "Endpoint",
    "ErrorRegistry",
    "IpcAddressInUseError",
    "IpcAuthError",
    "IpcClosedError",
    "IpcConfigError",
    "IpcConnectError",
    "IpcError",
    "IpcProtocolError",
    "IpcServer",
    "RemoteError",
    "RpcClient",
    "RpcConnection",
    "RpcError",
    "RpcInvalidParamsError",
    "RpcMethodNotFoundError",
    "RpcService",
    "RpcTimeoutError",
    "StreamCreditError",
    "StreamError",
    "StreamKind",
    "StreamMessage",
    "StreamReceiver",
    "StreamSender",
    "StreamService",
    "StreamWindow",
    "Wire",
    "connect_channel",
    "connect_rpc",
    "connect_stream",
    "decode_active_stream",
    "decode_camera_caps",
    "decode_camera_info",
    "decode_exception",
    "decode_json",
    "decode_recovery_level",
    "decode_roi",
    "decode_stream_config",
    "encode_active_stream",
    "encode_camera_caps",
    "encode_camera_info",
    "encode_exception",
    "encode_json",
    "encode_roi",
    "encode_stream_config",
    "load_connection_key",
]
