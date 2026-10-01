"""The middleware of the web process: the host check, the headers, the body limit, and compression.

Each class is a plain ASGI middleware. All of them leave the lifespan traffic alone. The last three
leave WebSocket traffic alone too.

- `AllowedHosts` is the outermost layer. It refuses an HTTP request or a WebSocket handshake whose
  `Host` is not an allowed host, and a handshake whose `Origin` is not. It never looks at the
  token, so it does not change who may read or send commands.
- `SecurityHeaders` adds the headers that tell a browser to treat the UI strictly (the opener
  policy only on a secure origin, because a browser ignores it elsewhere and logs an error), and
  the default `Cache-Control` of each part of the site. An API response is `no-store`, because it
  is live data. A static file is `no-cache`, which makes the browser ask again and accept a `304`
  answer, so an update of the UI shows at once and a visit costs almost no data. A route that sets
  its own `Cache-Control` (an image, which never changes) keeps it.
- `BodyLimit` refuses a request body above the limit, by its `Content-Length` and while it streams.
- `SelectiveGZip` compresses the text responses and leaves the images alone.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from seeingmon.services.web.errors import ApiError, error_body
from seeingmon.services.web.hosts import host_of_header, host_of_origin, is_trustworthy_origin

_log = logging.getLogger(__name__)

API_PREFIX = "/api/"

# The UI loads only its own files, and it draws on canvas elements and blob images. A WebSocket
# to the same host is allowed for the live view.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self' ws: wss:; object-src 'none'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("x-content-type-options", "nosniff"),
    ("x-frame-options", "DENY"),
    ("referrer-policy", "no-referrer"),
    ("cross-origin-resource-policy", "same-origin"),
    ("permissions-policy", "camera=(), microphone=(), geolocation=()"),
    ("content-security-policy", CONTENT_SECURITY_POLICY),
)

# A browser ignores these headers on an origin that is not secure, and it logs an error for each one
# on each page load. A LAN or VPN deployment serves plain HTTP, so the server sends them only on a
# secure origin: https, or a loopback address (see `is_trustworthy_origin`).
SECURE_ORIGIN_HEADERS: tuple[tuple[str, str], ...] = (
    ("cross-origin-opener-policy", "same-origin"),
)


def security_headers(scope: Scope) -> tuple[tuple[str, str], ...]:
    """The security headers for a request: the opener policy goes out only on a secure origin.

    The origin is the scheme of the request and its `Host`. `X-Forwarded-Proto` does not count,
    because the server does not sit behind a proxy that it trusts.
    """
    host = host_of_header(Headers(scope=scope).get("host", ""))
    if is_trustworthy_origin(host, str(scope.get("scheme", "http"))):
        return (*SECURITY_HEADERS, *SECURE_ORIGIN_HEADERS)
    return SECURITY_HEADERS


HOST_CODE = "host_not_allowed"
HOST_MESSAGE = (
    "The Host of this request is not allowed. Add the name or address that the client uses to "
    "allowed_hosts in the [web] section of the configuration."
)
ORIGIN_CODE = "origin_not_allowed"
ORIGIN_MESSAGE = (
    "The Origin of this WebSocket request is not allowed. The host of the page must be in "
    "allowed_hosts in the [web] section of the configuration."
)
WS_CLOSE_POLICY = 1008
MAX_LOGGED_HOSTS = 64
MAX_LOGGED_CHARS = 100


def _escape(char: str) -> str:
    if " " <= char <= "~":
        return char
    if ord(char) < 0x80:
        return f"\\x{ord(char):02x}"
    return char.encode("ascii", "backslashreplace").decode("ascii")


def printable(text: str, limit: int = MAX_LOGGED_CHARS) -> str:
    """Make header text safe for a log line: escape control and non-ASCII characters, and cut it.

    The result has at most `limit` characters, and a cut ends with `...`. A client chooses the
    text, so it must not be able to forge a line of the log or fill the log with one value.
    """
    escaped = "".join(_escape(char) for char in text[:limit])
    if len(text) <= limit and len(escaped) <= limit:
        return escaped
    return escaped[: limit - 3] + "..."


class AllowedHosts:
    """Refuse a request whose `Host` is not allowed, and a WebSocket whose `Origin` is not.

    `allowed` returns the set of allowed hosts in canonical form (see
    `seeingmon.services.web.hosts`). It is a function, so that the set follows the context of the
    app. The check covers every HTTP request (the API, the UI, and the static files) and every
    WebSocket handshake. A request with no `Host`, with more than one, or with a malformed one is
    refused too. A handshake without an `Origin` passes, because a client that is not a browser
    sends none.

    An HTTP request gets `400` with the JSON error of the API. A WebSocket handshake gets the same
    answer as an HTTP response (`403` for the origin) when the server supports it, and a close with
    code 1008 otherwise. The message names the setting `allowed_hosts` and never lists the allowed
    hosts. The middleware logs each distinct refused value once, up to `MAX_LOGGED_HOSTS` values.
    """

    def __init__(self, app: ASGIApp, allowed: Callable[[], frozenset[str]]) -> None:
        self.app = app
        self._allowed = allowed
        self._logged: set[str] = set()
        self._log_closed = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        allowed = self._allowed()
        hosts = headers.getlist("host")
        host = host_of_header(hosts[0]) if len(hosts) == 1 else None
        if host is None or host not in allowed:
            self._log_refusal("Host", hosts)
            await self._refuse(scope, receive, send, 400, HOST_CODE, HOST_MESSAGE)
            return
        if kind == "websocket":
            origins = headers.getlist("origin")
            if origins:
                origin = host_of_origin(origins[0]) if len(origins) == 1 else None
                if origin is None or origin not in allowed:
                    self._log_refusal("Origin", origins)
                    await self._refuse(scope, receive, send, 403, ORIGIN_CODE, ORIGIN_MESSAGE)
                    return
        await self.app(scope, receive, send)

    def _log_refusal(self, header: str, values: list[str]) -> None:
        if len(values) == 1:
            shown = f"'{printable(values[0])}'"
        else:
            shown = "missing" if not values else f"repeated {len(values)} times"
        key = f"{header} {shown}"
        if key in self._logged:
            return
        if len(self._logged) >= MAX_LOGGED_HOSTS:
            if not self._log_closed:
                self._log_closed = True
                _log.warning(
                    "refused more than %d distinct hosts: the log stops listing them",
                    MAX_LOGGED_HOSTS,
                )
            return
        self._logged.add(key)
        _log.warning(
            "refused a request with the %s %s: to serve it, add its host to allowed_hosts in the "
            "[web] section",
            header,
            shown,
        )

    @staticmethod
    async def _refuse(
        scope: Scope, receive: Receive, send: Send, status: int, code: str, message: str
    ) -> None:
        websocket = scope["type"] == "websocket"
        if websocket and "websocket.http.response" not in (scope.get("extensions") or {}):
            await send({"type": "websocket.close", "code": WS_CLOSE_POLICY})
            return
        headers = {**dict(security_headers(scope)), "cache-control": "no-store"}
        if not websocket:
            headers["connection"] = "close"
        response = JSONResponse(error_body(code, message), status_code=status, headers=headers)
        await response(scope, receive, send)


class SecurityHeaders:
    """Add the security headers and the default `Cache-Control` to every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        default_cache = "no-store" if scope["path"].startswith(API_PREFIX) else "no-cache"
        added = security_headers(scope)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in added:
                    headers.setdefault(name, value)
                headers.setdefault("cache-control", default_cache)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodyLimit:
    """Refuse a request body that is larger than `max_bytes` with a 413 answer."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._refuse(send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise ApiError(413)
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _refuse(send: Send) -> None:
        body = json.dumps(
            error_body("payload_too_large", "The request body is too large.")
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class SelectiveGZip:
    """Compress responses with gzip, except below the path prefixes in `skip_prefixes`."""

    def __init__(self, app: ASGIApp, skip_prefixes: tuple[str, ...]) -> None:
        self.app = app
        self.skip_prefixes = skip_prefixes
        self._gzip = GZipMiddleware(app, minimum_size=1024, compresslevel=4)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not scope["path"].startswith(self.skip_prefixes):
            await self._gzip(scope, receive, send)
        else:
            await self.app(scope, receive, send)
