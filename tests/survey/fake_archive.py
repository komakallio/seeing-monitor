"""A local HTTP server that stands in for the Gaia and VizieR TAP services in tests.

The answers are small tables in the shape of the real services. The server speaks the
protocol that the catalog build uses: an asynchronous job at `/gaia/async` (a redirect to the
job, a phase that changes after a few polls, and a result), and a synchronous query at
`/vizier/sync`. No test touches the real network.
"""

from __future__ import annotations

import threading
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType

GAIA_HEADER = "source_id,ra,dec,pmra,pmdec,parallax,phot_g_mean_mag,bp_rp"
TYCHO_HEADER = "TYC1,TYC2,TYC3,ra_mean,dec_mean,ra_obs,dec_obs,pmra,pmdec,bt_mag,vt_mag,hip"


@dataclass
class ArchiveScript:
    """What the fake archive answers. `requests` records every request it received."""

    gaia_csv: str
    tycho_csv: str
    polls_before_completed: int = 2
    final_phase: str = "COMPLETED"  # "ERROR" makes the job fail after the polls
    error_text: str = "ADQL syntax error near the cap"
    redirect: bool = True  # False: answer the job request with a UWS job document
    fail_phase_requests: int = 0  # answer HTTP 503 to this many phase requests first
    requests: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    phase_requests: int = 0


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:  # silence the test output
        return

    def _send(self, status: int, body: str = "", headers: dict[str, str] | None = None) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=UTF-8")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return {key: values[0] for key, values in urllib.parse.parse_qs(raw).items()}

    def do_POST(self) -> None:
        script = self.server.script
        form = self._form()
        script.requests.append(("POST", self.path, form))
        if self.path == "/gaia/async":
            if script.redirect:
                self._send(303, "", {"Location": "/gaia/async/job42"})
            else:
                document = (
                    '<?xml version="1.0"?><uws:job xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0">'
                    "<uws:jobId><![CDATA[job42]]></uws:jobId></uws:job>"
                )
                self._send(200, document)
        elif self.path == "/vizier/sync":
            self._send(200, script.tycho_csv)
        else:
            self._send(404, "unknown path")

    def do_GET(self) -> None:
        script = self.server.script
        script.requests.append(("GET", self.path, {}))
        if self.path == "/gaia/async/job42/phase":
            script.phase_requests += 1
            if script.phase_requests <= script.fail_phase_requests:
                self._send(503, "busy")
            elif (
                script.phase_requests - script.fail_phase_requests <= script.polls_before_completed
            ):
                self._send(200, "EXECUTING")
            else:
                self._send(200, script.final_phase)
        elif self.path == "/gaia/async/job42/results/result":
            self._send(200, script.gaia_csv)
        elif self.path == "/gaia/async/job42/error":
            self._send(200, script.error_text)
        else:
            self._send(404, "unknown path")


class _Server(ThreadingHTTPServer):
    script: ArchiveScript
    daemon_threads = True


class FakeArchive:
    """Run the fake archive on a free local port while the `with` block lasts."""

    def __init__(self, script: ArchiveScript) -> None:
        self.script = script
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.script = script
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    @property
    def gaia_url(self) -> str:
        return self.base_url + "/gaia"

    @property
    def vizier_url(self) -> str:
        return self.base_url + "/vizier"

    def __enter__(self) -> FakeArchive:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
