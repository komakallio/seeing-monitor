"""Exceptions of the connection layer.

Every failure of the layer is an `IpcError`. Code that talks to a peer catches the specific
class it can act on: `IpcConnectError` when nobody listens, `IpcAuthError` for a wrong or
missing key, `IpcClosedError` when the peer is gone, and `RpcTimeoutError` when a call takes
too long. `RemoteError` stands for an exception that the peer raised and that the caller did
not register (see `seeingmon.services.ipc.codec.ErrorRegistry`).
"""

from __future__ import annotations


class IpcError(Exception):
    """Base class for every failure of the connection layer."""


class IpcConfigError(IpcError, ValueError):
    """An address or a key is not usable. The message never contains the key."""


class IpcConnectError(IpcError):
    """Nobody answers at the address: the process is not running, or the address is wrong."""


class IpcAddressInUseError(IpcError):
    """Another process already listens at the address."""


class IpcAuthError(IpcError):
    """The peer did not prove that it holds the connection key, or rejected our proof."""


class IpcProtocolError(IpcError):
    """The peer sent something that the protocol does not allow."""


class IpcClosedError(IpcError):
    """The connection is closed, because we closed it or because the peer went away."""


class RpcError(IpcError):
    """Base class for failures of the request-response layer."""


class RpcTimeoutError(RpcError):
    """A call got no answer within its timeout. The connection stays usable."""


class RpcMethodNotFoundError(RpcError):
    """The peer has no method of that name."""


class RpcInvalidParamsError(RpcError, ValueError):
    """The peer rejected the parameters of a call."""


class RemoteError(RpcError):
    """The peer raised an exception that the caller does not know.

    `remote_type` is the name of the peer's exception class, and the message is its text.
    """

    def __init__(self, remote_type: str, message: str) -> None:
        super().__init__(f"{remote_type}: {message}" if message else remote_type)
        self.remote_type = remote_type
        self.remote_message = message


class StreamError(IpcError):
    """Base class for failures of the stream channel."""


class StreamCreditError(StreamError):
    """A sender tried to send without credit. Check `has_credit` first."""
