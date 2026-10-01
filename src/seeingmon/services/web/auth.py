"""The token hash, the bearer-token check, and the per-client rate limiter of the web process.

**Token hash format.** The configuration stores only a hash of the API token. The hash is a
string in the PHC style, made by `hashlib.scrypt` from the standard library:

    $scrypt$ln=15,r=8,p=1$<salt>$<digest>

`ln` is the base-2 logarithm of the CPU and memory cost N, `r` is the block size, and `p` is the
parallelism. The salt and the digest are standard base64 text without the `=` padding. A new
hash uses 16 random bytes of salt, N = 2^15 (32 MiB of memory), r = 8, p = 1, and a 32-byte
digest. The hash records its own parameters, so you can raise the cost later without breaking an
old hash. `seeingmon web hash-token` creates one. The token itself is random text that you keep in
a password manager. Generate it with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

**Checks.** A client sends the token as `Authorization: Bearer <token>`. `TokenVerifier` hashes
the candidate with the stored salt and compares the digests with `hmac.compare_digest`, so the
comparison takes the same time whatever the candidate is. Each check runs scrypt once, one check at
a time, so a flood of requests cannot use more than one hash worth of memory. A token that passed
stays in a small in-memory cache for a few minutes, so a page that sends many requests does not
pay for scrypt on each one. A failed check is never cached.

**Rate limits.** `RateLimiter` counts events per client in a sliding window. The web process uses
one limiter for commands and one for failed token checks. It reads the time from a `Clock`, so
tests run without waiting.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import math
import re
import secrets
import threading
from collections import deque
from dataclasses import dataclass

from seeingmon.clock import NS_PER_S, Clock

MIB = 1024 * 1024
SCRYPT_TAG = "scrypt"
SALT_BYTES = 16
DIGEST_BYTES = 32
MAX_TOKEN_CHARS = 256
MAX_MEMORY_BYTES = 256 * MIB

_BEARER = re.compile(r"(?i:bearer) +([A-Za-z0-9._~+/-]+=*)")
_PARAMS = re.compile(r"ln=([0-9]{1,2}),r=([0-9]{1,2}),p=([0-9]{1,2})")
_BASE64 = re.compile(r"[A-Za-z0-9+/]+")


class AuthConfigError(ValueError):
    """The hash of the token is malformed. The message never shows the hash."""


@dataclass(frozen=True, slots=True)
class ScryptParams:
    """The cost parameters of scrypt. The defaults suit a Raspberry Pi 4."""

    ln: int = 15
    r: int = 8
    p: int = 1

    def __post_init__(self) -> None:
        if not 4 <= self.ln <= 20 or not 1 <= self.r <= 32 or not 1 <= self.p <= 16:
            raise AuthConfigError("the scrypt parameters are outside the supported range")
        if self.memory_bytes > MAX_MEMORY_BYTES:
            raise AuthConfigError("the scrypt parameters need too much memory")

    @property
    def n(self) -> int:
        """The cost N, which is 2 to the power `ln`."""
        return 1 << self.ln

    @property
    def memory_bytes(self) -> int:
        """The memory that one hash needs, as OpenSSL counts it."""
        return 128 * self.r * (self.n + self.p + 2)


@dataclass(frozen=True, slots=True)
class TokenHash:
    """A parsed hash of a token: its parameters, its salt, and its digest."""

    params: ScryptParams
    salt: bytes
    digest: bytes


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    if not _BASE64.fullmatch(text):
        raise AuthConfigError("the token hash is not in the documented format")
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)
    except binascii.Error:
        raise AuthConfigError("the token hash is not in the documented format") from None


def _derive(token: bytes, salt: bytes, params: ScryptParams, length: int) -> bytes:
    return hashlib.scrypt(
        token,
        salt=salt,
        n=params.n,
        r=params.r,
        p=params.p,
        maxmem=params.memory_bytes * 2 + MIB,
        dklen=length,
    )


def hash_token(token: str, *, params: ScryptParams | None = None, salt: bytes | None = None) -> str:
    """Hash a token for the configuration. Each call draws a new random salt.

    Raises `AuthConfigError` when the token is empty or longer than `MAX_TOKEN_CHARS`.
    """
    if not token or len(token) > MAX_TOKEN_CHARS:
        raise AuthConfigError(f"a token has 1 to {MAX_TOKEN_CHARS} characters")
    chosen = params or ScryptParams()
    salt_bytes = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    digest = _derive(token.encode("utf-8"), salt_bytes, chosen, DIGEST_BYTES)
    return (
        f"${SCRYPT_TAG}$ln={chosen.ln},r={chosen.r},p={chosen.p}"
        f"${_b64encode(salt_bytes)}${_b64encode(digest)}"
    )


def parse_token_hash(text: str) -> TokenHash:
    """Parse the hash that `hash_token` made. Raises `AuthConfigError` for anything else."""
    parts = text.strip().split("$")
    if len(parts) != 5 or parts[0] != "" or parts[1] != SCRYPT_TAG:
        raise AuthConfigError(
            "the token hash is not in the documented format: create one with "
            "`seeingmon web hash-token`"
        )
    match = _PARAMS.fullmatch(parts[2])
    if match is None:
        raise AuthConfigError("the token hash is not in the documented format")
    params = ScryptParams(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    salt, digest = _b64decode(parts[3]), _b64decode(parts[4])
    if not 8 <= len(salt) <= 64 or not 16 <= len(digest) <= 64:
        raise AuthConfigError("the token hash is not in the documented format")
    return TokenHash(params, salt, digest)


def verify_token(token: str, stored: str) -> bool:
    """Whether `token` matches a stored hash. Use `TokenVerifier` on a request path."""
    parsed = parse_token_hash(stored)
    try:
        candidate = token.encode("utf-8")
    except UnicodeEncodeError:
        return False
    derived = _derive(candidate, parsed.salt, parsed.params, len(parsed.digest))
    return hmac.compare_digest(derived, parsed.digest)


def bearer_token(header: str | None) -> str | None:
    """The token of an `Authorization: Bearer <token>` header, or `None` for anything else.

    `None` covers a missing header, another scheme, a token with characters that a bearer token
    never holds, and a token over `MAX_TOKEN_CHARS`.
    """
    if header is None or len(header) > MAX_TOKEN_CHARS + 32:
        return None
    match = _BEARER.fullmatch(header.strip())
    if match is None or len(match.group(1)) > MAX_TOKEN_CHARS:
        return None
    return match.group(1)


class TokenVerifier:
    """Check candidate tokens against the stored hash. Safe to call from many threads.

    With no hash, `enabled` is `False` and no candidate passes. The token that last passed is
    remembered for `cache_ttl_s` seconds as a keyed digest, with a key that the process draws at
    start. The verifier never keeps the token itself, and only the right token can pass, so one
    entry is all it needs.
    """

    def __init__(self, token_hash: str | None, clock: Clock, *, cache_ttl_s: float = 300.0) -> None:
        self._hash = None if token_hash is None else parse_token_hash(token_hash)
        self._clock = clock
        self._ttl_ns = round(cache_ttl_s * NS_PER_S)
        self._cache_key = secrets.token_bytes(32)
        self._passed_key = bytes(32)
        self._passed_until_ns = 0
        self._lock = threading.Lock()
        self._work = threading.Semaphore(1)

    @property
    def enabled(self) -> bool:
        """Whether a hash is configured. Without one, the API refuses every command."""
        return self._hash is not None

    def verify(self, token: str) -> bool:
        """Whether the candidate is the token. A wrong candidate costs one scrypt run."""
        stored = self._hash
        if stored is None or not token or len(token) > MAX_TOKEN_CHARS:
            return False
        try:
            candidate = token.encode("utf-8")
        except UnicodeEncodeError:
            return False
        key = hmac.new(self._cache_key, candidate, hashlib.sha256).digest()
        if self._remembered(key):
            return True
        with self._work:
            derived = _derive(candidate, stored.salt, stored.params, len(stored.digest))
        matched = hmac.compare_digest(derived, stored.digest)
        if matched:
            with self._lock:
                self._passed_key = key
                self._passed_until_ns = self._clock.monotonic_ns() + self._ttl_ns
        return matched

    def _remembered(self, key: bytes) -> bool:
        now_ns = self._clock.monotonic_ns()
        with self._lock:
            same = hmac.compare_digest(self._passed_key, key)
            return same and self._passed_until_ns > now_ns


class RateLimiter:
    """Allow `limit` events per `window_s` for each client, counted in a sliding window.

    `acquire` counts an event when the client is under its limit. `check` and `record` split the
    two steps, for a limit that counts only some events (the failed token checks). The limiter
    tracks at most `max_clients` clients and forgets the least recently active one first.
    """

    def __init__(
        self, clock: Clock, *, limit: int, window_s: float, max_clients: int = 1024
    ) -> None:
        if limit < 1 or window_s <= 0 or max_clients < 1:
            raise ValueError("limit, window_s, and max_clients must be positive")
        self._clock = clock
        self._limit = limit
        self._window_ns = round(window_s * NS_PER_S)
        self._max_clients = max_clients
        self._events: dict[str, deque[int]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now_ns: int) -> deque[int]:
        events = self._events.get(key)
        if events is None:
            events = deque()
        floor = now_ns - self._window_ns
        while events and events[0] <= floor:
            events.popleft()
        return events

    def _retry_after_s(self, events: deque[int], now_ns: int) -> int | None:
        if len(events) < self._limit:
            return None
        wait_ns = events[0] + self._window_ns - now_ns
        return max(1, math.ceil(wait_ns / NS_PER_S))

    def check(self, key: str) -> int | None:
        """Seconds until the client may act again, or `None` when it is under its limit."""
        now_ns = self._clock.monotonic_ns()
        with self._lock:
            return self._retry_after_s(self._prune(key, now_ns), now_ns)

    def record(self, key: str) -> None:
        """Count one event for the client."""
        now_ns = self._clock.monotonic_ns()
        with self._lock:
            self._store(key, now_ns, self._prune(key, now_ns))

    def acquire(self, key: str) -> int | None:
        """Count an event if the client is under its limit. Otherwise return the seconds to wait."""
        now_ns = self._clock.monotonic_ns()
        with self._lock:
            events = self._prune(key, now_ns)
            wait = self._retry_after_s(events, now_ns)
            if wait is None:
                self._store(key, now_ns, events)
            return wait

    def _store(self, key: str, now_ns: int, events: deque[int]) -> None:
        events.append(now_ns)
        self._events.pop(key, None)
        self._events[key] = events  # the newest client sits last, so eviction starts at the front
        if len(self._events) > self._max_clients:
            floor = now_ns - self._window_ns
            for name in [name for name, items in self._events.items() if items[-1] <= floor]:
                del self._events[name]
            while len(self._events) > self._max_clients:
                del self._events[next(iter(self._events))]

    @property
    def clients(self) -> int:
        """The number of clients that the limiter tracks."""
        with self._lock:
            return len(self._events)
