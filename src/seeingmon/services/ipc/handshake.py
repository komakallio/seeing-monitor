"""The proof that both ends hold the connection key.

A new connection runs a challenge and response before it exchanges anything else. The key
itself never crosses the connection, and both ends prove that they hold it:

1. The server sends a random challenge.
2. The client answers with an HMAC-SHA256 of both nonces (labeled `client`) and its own
   random challenge.
3. The server checks the answer in constant time, and then answers with an HMAC of both nonces
   (labeled `server`). A server that rejects the client says so with one short message.
4. The client checks the server's proof.

The labels stop a reflected answer, and fresh nonces stop a replayed one. Every message is
short and has a fixed length, and every wait has a timeout, so a peer that stays silent or
sends garbage costs the server a few hundred bytes and `timeout_s` seconds.

The messages use only `send_bytes` and `recv_bytes`, like the rest of the layer.
"""

from __future__ import annotations

import contextlib
import secrets

from seeingmon.services.ipc.errors import IpcAuthError, IpcClosedError, IpcProtocolError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.wire import Wire

MAGIC = b"SMC1"
NONCE_BYTES = 32
MAC_BYTES = 32
_CHALLENGE = MAGIC + b"\x01"
_ANSWER = MAGIC + b"\x02"
_REJECTED = MAGIC + b"\x00"
_WELCOME = MAGIC + b"\x03"
_HEADER = len(MAGIC) + 1
MAX_HANDSHAKE_BYTES = 128


def serve_handshake(wire: Wire, key: ConnectionKey, *, timeout_s: float) -> None:
    """Challenge the client. Raises `IpcAuthError` unless it proves that it holds the key."""
    server_nonce = secrets.token_bytes(NONCE_BYTES)
    try:
        wire.send(_CHALLENGE + server_nonce)
        answer = wire.recv(timeout_s, max_bytes=MAX_HANDSHAKE_BYTES)
    except (IpcClosedError, IpcProtocolError) as error:
        raise IpcAuthError("the client left during the handshake") from error
    if answer is None:
        raise IpcAuthError("the client did not answer the challenge in time")
    if len(answer) != _HEADER + MAC_BYTES + NONCE_BYTES or not answer.startswith(_ANSWER):
        raise IpcAuthError("the client sent a malformed answer")
    proof = answer[_HEADER : _HEADER + MAC_BYTES]
    client_nonce = answer[_HEADER + MAC_BYTES :]
    if not key.verify(proof, b"client", server_nonce, client_nonce):
        with contextlib.suppress(IpcClosedError):
            wire.send(_REJECTED)
        raise IpcAuthError("the client does not hold the connection key")
    try:
        wire.send(_WELCOME + key.sign(b"server", server_nonce, client_nonce))
    except IpcClosedError as error:
        raise IpcAuthError("the client left during the handshake") from error


def client_handshake(wire: Wire, key: ConnectionKey, *, timeout_s: float) -> None:
    """Answer the server's challenge. Raises `IpcAuthError` when the keys do not match."""
    try:
        challenge = wire.recv(timeout_s, max_bytes=MAX_HANDSHAKE_BYTES)
        if challenge is None:
            raise IpcAuthError("the server sent no challenge in time")
        if len(challenge) != _HEADER + NONCE_BYTES or not challenge.startswith(_CHALLENGE):
            raise IpcAuthError("the peer is not a seeingmon service")
        server_nonce = challenge[_HEADER:]
        client_nonce = secrets.token_bytes(NONCE_BYTES)
        wire.send(_ANSWER + key.sign(b"client", server_nonce, client_nonce) + client_nonce)
        reply = wire.recv(timeout_s, max_bytes=MAX_HANDSHAKE_BYTES)
    except (IpcClosedError, IpcProtocolError) as error:
        # A server that rejects the key may close before we read its message.
        raise IpcAuthError(
            "the server closed the connection during the handshake; check the connection key"
        ) from error
    if reply is None:
        raise IpcAuthError("the server did not answer in time")
    if reply == _REJECTED:
        raise IpcAuthError("the server rejected the connection key")
    if (
        len(reply) != _HEADER + MAC_BYTES
        or not reply.startswith(_WELCOME)
        or not key.verify(reply[_HEADER:], b"server", server_nonce, client_nonce)
    ):
        raise IpcAuthError("the server did not prove that it holds the connection key")
