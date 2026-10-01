"""The shared secret of the local connections.

The processes of one installation prove to each other that they hold the same key before they
exchange anything (see `seeingmon.services.ipc.handshake`). The key never lives in the
repository. `load_connection_key` reads it from the first source that has one:

1. The configured value (`connection_key` in `[services]`, or the environment variable
   `SEEINGMON_SERVICES__CONNECTION_KEY`).
2. A systemd credential: the file `$CREDENTIALS_DIRECTORY/<name>`.
3. A file whose path the configuration names (`connection_key_file`).

A key is text of 16 to 1,024 characters, and leading and trailing whitespace does not count,
so a key file may end with a newline. Generate one with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`.

A `ConnectionKey` hides its value from `repr`, `str`, and tracebacks.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping
from pathlib import Path

from seeingmon.services.ipc.errors import IpcConfigError

MIN_KEY_CHARS = 16
MAX_KEY_CHARS = 1024
_MAX_KEY_FILE_BYTES = 8 * 1024


class ConnectionKey:
    """A shared secret. It signs and verifies short messages with HMAC-SHA256."""

    __slots__ = ("_secret",)

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes):
            raise IpcConfigError("a connection key is built from bytes")
        if not MIN_KEY_CHARS <= len(secret) <= MAX_KEY_CHARS * 4:
            raise IpcConfigError(
                f"a connection key has {MIN_KEY_CHARS} to {MAX_KEY_CHARS} characters"
            )
        self._secret = secret

    @classmethod
    def from_text(cls, text: str) -> ConnectionKey:
        """Build a key from text. Whitespace at both ends is not part of the key."""
        stripped = text.strip()
        if not MIN_KEY_CHARS <= len(stripped) <= MAX_KEY_CHARS:
            raise IpcConfigError(
                f"a connection key has {MIN_KEY_CHARS} to {MAX_KEY_CHARS} characters"
            )
        return cls(stripped.encode("utf-8"))

    def sign(self, *parts: bytes) -> bytes:
        """The HMAC-SHA256 of the parts. Each part is length-prefixed, so none can shift."""
        mac = hmac.new(self._secret, digestmod=hashlib.sha256)
        for part in parts:
            mac.update(len(part).to_bytes(4, "big"))
            mac.update(part)
        return mac.digest()

    def verify(self, signature: bytes, *parts: bytes) -> bool:
        """Whether `signature` is the signature of the parts. The comparison takes constant time."""
        return hmac.compare_digest(signature, self.sign(*parts))

    def __repr__(self) -> str:
        return "ConnectionKey(<hidden>)"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConnectionKey) and hmac.compare_digest(self._secret, other._secret)

    def __hash__(self) -> int:
        return hash(hashlib.sha256(self._secret).digest())


def _read_key_file(path: Path, description: str) -> ConnectionKey:
    try:
        if path.stat().st_size > _MAX_KEY_FILE_BYTES:
            raise IpcConfigError(f"the {description} is too large to be a key")
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise IpcConfigError(
            f"cannot read the {description}: {error.strerror or type(error).__name__}"
        ) from None
    except UnicodeDecodeError:
        raise IpcConfigError(f"the {description} is not UTF-8 text") from None
    return ConnectionKey.from_text(text)


def load_connection_key(
    *,
    value: str | None = None,
    credential: str | None = None,
    file: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ConnectionKey:
    """Find the connection key in the configured value, a systemd credential, or a file.

    `credential` is the name of a systemd credential, and `env` (default `os.environ`) supplies
    `CREDENTIALS_DIRECTORY`. Raises `IpcConfigError` when no source has a key, or when the
    key is unusable. The message names the sources and never shows a key.
    """
    environment = os.environ if env is None else env
    if value is not None and value.strip():
        return ConnectionKey.from_text(value)
    directory = environment.get("CREDENTIALS_DIRECTORY", "")
    if credential and directory:
        candidate = Path(directory) / credential
        if candidate.is_file():
            return _read_key_file(candidate, "systemd credential")
    if file and file.strip():
        return _read_key_file(Path(file.strip()), "key file")
    raise IpcConfigError(
        "no connection key is configured: set connection_key in [services] "
        "(or SEEINGMON_SERVICES__CONNECTION_KEY), provide a systemd credential, or set "
        "connection_key_file"
    )
