"""A local HTTP server that stands in for InfluxDB. It records requests and simulates faults.

Script the replies with `server.script(...)`: the server applies one behavior to each request, in
order, and then falls back to `server.default` (204 No Content). A behavior is one of:

- `Reply(status, body)`: answer with a status and a body.
- `Hang(seconds)`: wait, then answer 204. A client with a shorter timeout gives up first.
- `Drop()`: close the connection without a reply, as a crashed server does.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Self
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class Reply:
    status: int = 204
    body: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Hang:
    seconds: float


@dataclass(frozen=True)
class Drop:
    pass


Behavior = Reply | Hang | Drop


@dataclass(frozen=True)
class Recorded:
    """One request that the server received."""

    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]  # the names in lowercase
    body: str


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request: object, client_address: object) -> None:
        pass  # a client that gave up and left is not an error of the test


class FakeInfluxServer:
    """Use it as a context manager. `url` is the address to give the sink as its endpoint."""

    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self.default: Behavior = Reply(204)
        self._script: deque[Behavior] = deque()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def handle_request_body(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8") if length else ""
                parts = urlsplit(self.path)
                outer.requests.append(
                    Recorded(
                        self.command,
                        parts.path,
                        parse_qs(parts.query),
                        {name.lower(): value for name, value in self.headers.items()},
                        body,
                    )
                )
                outer._act(self, outer._next())

            do_POST = handle_request_body  # noqa: N815
            do_GET = handle_request_body  # noqa: N815

        self._httpd = _Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        self._httpd.serve_forever(poll_interval=0.01)  # a short poll makes `shutdown` quick

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def script(self, *behaviors: Behavior) -> None:
        with self._lock:
            self._script.extend(behaviors)

    def _next(self) -> Behavior:
        with self._lock:
            return self._script.popleft() if self._script else self.default

    def _act(self, handler: BaseHTTPRequestHandler, behavior: Behavior) -> None:
        if isinstance(behavior, Drop):
            handler.close_connection = True  # no reply: the client sees the connection close
            return
        if isinstance(behavior, Hang):
            self._stop.wait(behavior.seconds)
            behavior = Reply(204)
        payload = behavior.body.encode("utf-8")
        handler.send_response(behavior.status)
        for name, value in behavior.headers.items():
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(payload)
        handler.close_connection = True

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=10)
