"""The middleware of the web process: security headers, cache headers, body limit, compression.

Each class is a plain ASGI middleware, so it works for HTTP and leaves WebSocket traffic alone.

- `SecurityHeaders` adds the headers that tell a browser to treat the UI strictly, and the default
  `Cache-Control` of each part of the site. An API response is `no-store`, because it is live data.
  A static file is `no-cache`, which makes the browser ask again and accept a `304` answer, so an
  update of the UI shows at once and a visit costs almost no data. A route that sets its own
  `Cache-Control` (an image, which never changes) keeps it.
- `BodyLimit` refuses a request body above the limit, by its `Content-Length` and while it streams.
- `SelectiveGZip` compresses the text responses and leaves the images alone.
"""

from __future__ import annotations

import json

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from seeingmon.services.web.errors import ApiError, error_body

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
    ("cross-origin-opener-policy", "same-origin"),
    ("cross-origin-resource-policy", "same-origin"),
    ("permissions-policy", "camera=(), microphone=(), geolocation=()"),
    ("content-security-policy", CONTENT_SECURITY_POLICY),
)


class SecurityHeaders:
    """Add the security headers and the default `Cache-Control` to every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        default_cache = "no-store" if scope["path"].startswith(API_PREFIX) else "no-cache"

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS:
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
