"""The host rule: a request or a WebSocket handshake that names a host nobody allowed is refused.

The rule extends the access rule (deferred decision B7). It does not change who may read or send
commands, so these tests check hosts and nothing about tokens, except that a refused request never
reaches the token check.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.testclient import WebSocketDenialResponse
from starlette.types import Message, Scope

from seeingmon.config import load_config
from seeingmon.services.web import config as web_config
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.core_client import FakeCoreClient
from seeingmon.services.web.middleware import (
    MAX_LOGGED_CHARS,
    MAX_LOGGED_HOSTS,
    AllowedHosts,
    printable,
)
from seeingmon.store.db import Store
from tests.services.web.client import TestClient
from tests.services.web.helpers import bearer

API = "/api/v1"
STREAM = f"{API}/alignment/stream"
LISTED = ["pi.example", "192.0.2.10", "2001:db8::10"]
LOGGER = "seeingmon.services.web.middleware"
MDNS_NAME = "my-pi.local"  # repo-check: allow
MULTICAST_V6 = "ff02::1"  # repo-check: allow


def host(name: str) -> dict[str, str]:
    return {"host": name}


@pytest.fixture
def served(
    settings: WebSettings,
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
) -> Callable[..., TestClient]:
    """Open a client of a server whose `[web]` section has the given changes."""

    def build(**changes: Any) -> TestClient:
        chosen = WebSettings.model_validate({**settings.model_dump(), **changes})
        return open_client(make_app(settings=chosen))

    return build


@pytest.fixture
def listed(served: Callable[..., TestClient]) -> TestClient:
    """A client of a server that lists a name, an IPv4 address, and an IPv6 address."""
    return served(allowed_hosts=LISTED)


# --- Which hosts pass ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "localhost",
        "localhost:8080",
        "LOCALHOST:9",
        "localhost.",
        "127.0.0.1",
        "127.0.0.1:8080",
        "[::1]",
        "[::1]:8080",
    ],
)
def test_the_default_needs_no_list_for_the_loopback_names(client: TestClient, name: str) -> None:
    assert client.get(f"{API}/status", headers=host(name)).status_code == 200


def test_a_bind_address_and_an_extra_address_pass_without_a_list(
    served: Callable[..., TestClient],
) -> None:
    client = served(bind_address="192.0.2.5", extra_bind_addresses=["2001:db8::5", "192.0.2.6"])
    for name in ("192.0.2.5", "192.0.2.5:8080", "[2001:db8::5]:8080", "192.0.2.6:80"):
        assert client.get(f"{API}/status", headers=host(name)).status_code == 200, name
    assert client.get(f"{API}/status", headers=host("192.0.2.7")).status_code == 400


@pytest.mark.parametrize(
    "name",
    [
        "pi.example",
        "pi.example:8080",
        "PI.EXAMPLE",
        "Pi.Example.",
        "pi.example.:8080",
        "192.0.2.10",
        "192.0.2.10:8080",
        "[2001:db8::10]",
        "[2001:db8::10]:8080",
        "[2001:DB8:0::10]:8080",
        "[2001:0db8:0000:0000:0000:0000:0000:0010]:8080",
    ],
)
def test_a_listed_host_passes_in_every_spelling(listed: TestClient, name: str) -> None:
    assert listed.get(f"{API}/status", headers=host(name)).status_code == 200


@pytest.mark.parametrize(
    "name",
    [
        "evil.example",
        "pi.example.evil.example",
        "xpi.example",
        "pi.examples",
        "pi",
        "192.0.2.11",
        "192.0.2.1",
        "[2001:db8::11]:8080",
        "2001:db8::10",
        "pi.example:abc",
        "pi.example:70000",
        "pi.example@evil.example",
        "evil.example/pi.example",
        "pi.example evil.example",
        "pi.example:8080:8080",
        "127.1",
        "0.0.0.0",
        "[::]",
    ],
)
def test_another_host_is_refused(listed: TestClient, name: str) -> None:
    response = listed.get(f"{API}/status", headers=host(name))
    assert response.status_code == 400, name
    assert response.json()["error"]["code"] == "host_not_allowed"


@pytest.fixture
def everywhere(
    served: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch
) -> Callable[[], TestClient]:
    """A client of a server that listens on every IPv4 interface, and lists one name."""
    monkeypatch.setattr(web_config, "device_names", lambda: ("my-pi", MDNS_NAME))
    return lambda: served(bind_address="0.0.0.0", allowed_hosts=["pi.example"])


@pytest.mark.parametrize(
    "name",
    [
        "192.0.2.77",
        "192.0.2.77:8080",
        "198.51.100.2",
        "[2001:db8::77]:8080",
        "localhost",
        "pi.example",
        "my-pi",
        MDNS_NAME.upper() + ":8080",
    ],
)
def test_a_wildcard_bind_passes_any_address_and_the_names_it_knows(
    everywhere: Callable[[], TestClient], name: str
) -> None:
    assert everywhere().get(f"{API}/status", headers=host(name)).status_code == 200


@pytest.mark.parametrize(
    "name",
    [
        "evil.example",
        "other-pi",
        "my-pi.evil.example",
        "192.0.2.77.evil.example",
        "0.0.0.0",
        "0.0.0.0:8080",
        "[::]",
        "224.0.0.1",
        f"[{MULTICAST_V6}]",
        "127.1",
    ],
)
def test_a_wildcard_bind_still_refuses_other_names_and_unusable_addresses(
    everywhere: Callable[[], TestClient], name: str
) -> None:
    response = everywhere().get(f"{API}/status", headers=host(name))
    assert response.status_code == 400, name
    assert response.json()["error"]["code"] == "host_not_allowed"


def test_a_wildcard_bind_checks_the_origin_of_a_websocket_like_the_host(
    everywhere: Callable[[], TestClient],
) -> None:
    client = everywhere()
    headers = {"host": "192.0.2.77:8080", "origin": "http://192.0.2.77:8080"}
    with client.websocket_connect(STREAM, headers=headers):
        pass
    assert refused(client, host="192.0.2.77:8080", origin="http://evil.example").status_code == 403


def test_the_empty_host_and_a_second_host_header_are_refused(listed: TestClient) -> None:
    assert listed.get(f"{API}/status", headers=host("")).status_code == 400
    twice = [("host", "localhost"), ("host", "localhost")]
    assert listed.get(f"{API}/status", headers=twice).status_code == 400
    mixed = [("host", "localhost"), ("host", "evil.example")]
    assert listed.get(f"{API}/status", headers=mixed).status_code == 400


def test_a_forwarded_host_changes_nothing(client: TestClient) -> None:
    forwarded = {"x-forwarded-host": "localhost", "forwarded": "host=localhost"}
    assert (
        client.get(f"{API}/status", headers={**host("evil.example"), **forwarded}).status_code
        == 400
    )
    other = {"x-forwarded-host": "evil.example", "forwarded": "host=evil.example"}
    assert client.get(f"{API}/status", headers={**host("localhost"), **other}).status_code == 200


def test_the_allowed_set_follows_the_context_of_the_app(
    served: Callable[..., TestClient],
) -> None:
    first = served(allowed_hosts=["pi.example"])
    assert first.get(f"{API}/status", headers=host("pi.example")).status_code == 200
    second = served()
    assert second.get(f"{API}/status", headers=host("pi.example")).status_code == 400


# --- Where the rule applies ------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        f"{API}/status",
        f"{API}/seeing/latest",
        f"{API}/images/latest",
        f"{API}/openapi.json",
        f"{API}/config",
        f"{API}/nothing",
        f"{API}/alignment/frame",
        "/",
        "/index.html",
        "/missing-file.css",
        "/missing/sub/dir.js",
    ],
)
def test_an_unlisted_host_gets_400_on_an_api_path_a_ui_path_and_a_static_path(
    client: TestClient, path: str
) -> None:
    response = client.get(path, headers=host("evil.example"))
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "host_not_allowed"
    assert body["error"]["details"] is None
    assert "allowed_hosts" in body["error"]["message"]


def test_a_listed_host_reaches_the_ui_and_the_api(listed: TestClient) -> None:
    assert listed.get("/", headers=host("pi.example:8080")).status_code == 200
    assert listed.get(f"{API}/seeing/latest", headers=host("pi.example")).status_code == 200


def test_the_refusal_names_the_setting_and_lists_no_allowed_value(
    served: Callable[..., TestClient],
) -> None:
    client = served(
        bind_address="192.0.2.77",
        extra_bind_addresses=["192.0.2.78"],
        allowed_hosts=["private-name.example"],
    )
    response = client.get(f"{API}/status", headers=host("evil.example"))
    assert response.status_code == 400
    everything = response.text + str(dict(response.headers))
    assert "allowed_hosts" in everything
    for allowed in ("private-name", "192.0.2.77", "192.0.2.78", "localhost", "127.0.0.1"):
        assert allowed not in everything


def test_the_refusal_is_not_cached_and_carries_the_security_headers(client: TestClient) -> None:
    headers = client.get("/", headers=host("evil.example")).headers
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in headers["content-security-policy"]


def test_the_host_check_runs_before_every_other_check(client: TestClient) -> None:
    bad = host("evil.example")
    assert client.get(f"{API}/nothing", headers=bad).status_code == 400  # not 404
    assert client.delete(f"{API}/status", headers=bad).status_code == 400  # not 405
    assert client.get(f"{API}/seeing", params={"limit": 0}, headers=bad).status_code == 400  # 422
    assert client.post(f"{API}/commands/burst", json={}, headers=bad).status_code == 400  # 401
    right = {**bearer(), **bad}
    assert client.post(f"{API}/commands/burst", json={}, headers=right).status_code == 400
    huge = client.post(f"{API}/commands/burst", content=b"x" * 100_000, headers=bad)
    assert huge.status_code == 400  # not 413


def test_a_refused_request_reaches_neither_core_nor_the_rate_limits(
    client: TestClient, core: FakeCoreClient
) -> None:
    wrong = {"Authorization": "Bearer wrong", "host": "evil.example"}
    for _ in range(20):  # the test limits are 5 commands and 5 failures per window
        assert client.post(f"{API}/commands/burst", json={}, headers=wrong).status_code == 400
    assert core.submitted == []
    assert client.post(f"{API}/commands/burst", json={}, headers=bearer()).status_code == 200


def test_the_host_rule_does_not_change_who_may_read_or_send_commands(
    served: Callable[..., TestClient],
) -> None:
    open_reads = served(allowed_hosts=["pi.example"])
    pi = host("pi.example")
    assert open_reads.get(f"{API}/status", headers=pi).status_code == 200  # reads stay open
    assert open_reads.post(f"{API}/commands/burst", json={}, headers=pi).status_code == 401
    assert (
        open_reads.post(f"{API}/commands/burst", json={}, headers={**pi, **bearer()}).status_code
        == 200
    )
    locked = served(allowed_hosts=["pi.example"], require_token_for_reads=True)
    assert locked.get(f"{API}/status", headers=pi).status_code == 401
    assert locked.get(f"{API}/status", headers={**pi, **bearer()}).status_code == 200


# --- WebSocket handshakes --------------------------------------------------------------------


def refused(client: TestClient, **headers: str) -> WebSocketDenialResponse:
    """Open the live view with the given headers, and return the refusal."""
    with (
        pytest.raises(WebSocketDenialResponse) as raised,
        client.websocket_connect(STREAM, headers=dict(headers)),
    ):
        pass
    return raised.value


def test_a_websocket_with_an_unlisted_host_is_refused_with_the_error_of_the_api(
    client: TestClient,
) -> None:
    denial = refused(client, host="evil.example")
    assert denial.status_code == 400
    assert denial.json()["error"]["code"] == "host_not_allowed"
    assert "allowed_hosts" in denial.json()["error"]["message"]


def test_a_websocket_with_a_foreign_origin_is_refused(client: TestClient) -> None:
    denial = refused(client, origin="http://evil.example")
    assert denial.status_code == 403
    body = denial.json()
    assert body["error"]["code"] == "origin_not_allowed"
    assert "allowed_hosts" in body["error"]["message"]
    assert "localhost" not in denial.text


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "",
        "http://evil.example:8080",
        "http://localhost.evil.example",
        "http://evil.example/localhost",
        "http://localhost@evil.example",
        "https://[2001:db8::99]",
        "file://",
        "chrome-extension://abcdef",  # repo-check: allow
        "localhost",
        "http://",
    ],
)
def test_a_websocket_origin_without_an_allowed_host_is_refused(
    client: TestClient, origin: str
) -> None:
    assert refused(client, origin=origin).status_code == 403


@pytest.fixture
def quiet_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    settings: WebSettings,
) -> TestClient:
    """A client of a server whose live view is idle, and that lists `LISTED`."""
    chosen = WebSettings.model_validate({**settings.model_dump(), "allowed_hosts": LISTED})
    return open_client(make_app(settings=chosen, core=FakeCoreClient()))


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost",
        "http://localhost:8080",
        "https://LOCALHOST:8443",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
        "http://pi.example",
        "http://Pi.Example.:8080",
        "http://192.0.2.10:8080",
        "http://[2001:db8::10]:8080",
    ],
)
def test_a_websocket_origin_with_an_allowed_host_is_accepted(
    quiet_client: TestClient, origin: str
) -> None:
    # Entering the context is the handshake. A refusal raises before the body runs.
    with quiet_client.websocket_connect(STREAM, headers={"origin": origin}):
        pass


def test_a_websocket_without_an_origin_is_accepted(quiet_client: TestClient) -> None:
    with quiet_client.websocket_connect(STREAM) as session:
        assert session.receive_json() == {"type": "idle"}


def test_a_websocket_with_a_listed_host_is_accepted(quiet_client: TestClient) -> None:
    headers = {"host": "pi.example:8080", "origin": "http://pi.example:8080"}
    with quiet_client.websocket_connect(STREAM, headers=headers) as session:
        assert session.receive_json() == {"type": "idle"}


def test_a_websocket_refusal_does_not_reach_the_hub(client: TestClient) -> None:
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    refused(client, origin="http://evil.example")
    refused(client, host="evil.example")
    assert ctx.hub.viewers == 0


# --- The middleware by itself ----------------------------------------------------------------


class Probe:
    """An `AllowedHosts` over an app that counts the requests that reach it."""

    def __init__(self, allowed: frozenset[str] = frozenset({"localhost"})) -> None:
        self.reached: list[str] = []
        self.middleware = AllowedHosts(self.app, lambda: allowed)

    async def app(self, scope: Scope, receive: Any, send: Any) -> None:
        self.reached.append(scope["type"])

    def call(self, scope: Scope) -> list[Message]:
        sent: list[Message] = []

        async def receive() -> Message:
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            sent.append(message)

        asyncio.run(self.middleware(scope, receive, send))
        return sent


def scope_of(
    kind: str = "http",
    *,
    hosts: tuple[str, ...] = ("localhost",),
    origins: tuple[str, ...] = (),
    extensions: dict[str, Any] | None = None,
) -> Scope:
    headers = [(b"host", value.encode("latin-1")) for value in hosts]
    headers += [(b"origin", value.encode("latin-1")) for value in origins]
    scope: Scope = {"type": kind, "path": "/", "headers": headers}
    if extensions is not None:
        scope["extensions"] = extensions
    return scope


def test_the_lifespan_passes_through() -> None:
    probe = Probe()
    assert probe.call({"type": "lifespan"}) == []
    assert probe.reached == ["lifespan"]


def test_a_request_with_no_host_header_is_refused() -> None:
    probe = Probe()
    sent = probe.call(scope_of(hosts=()))
    assert sent[0]["status"] == 400
    assert probe.reached == []


def test_a_request_with_two_host_headers_is_refused() -> None:
    probe = Probe()
    assert probe.call(scope_of(hosts=("localhost", "localhost")))[0]["status"] == 400
    assert probe.reached == []


def test_a_websocket_with_two_origin_headers_is_refused() -> None:
    probe = Probe()
    origins = ("http://localhost", "http://localhost")
    scope = scope_of("websocket", origins=origins, extensions={"websocket.http.response": {}})
    start, _body = probe.call(scope)
    assert start["status"] == 403
    assert probe.reached == []


def test_an_http_request_ignores_the_origin_header() -> None:
    probe = Probe()
    probe.call(scope_of(origins=("http://evil.example",)))
    assert probe.reached == ["http"]


def test_a_server_without_the_denial_extension_closes_the_websocket_with_code_1008() -> None:
    probe = Probe()
    assert probe.call(scope_of("websocket", hosts=("evil.example",))) == [
        {"type": "websocket.close", "code": 1008}
    ]
    assert probe.call(scope_of("websocket", origins=("http://evil.example",))) == [
        {"type": "websocket.close", "code": 1008}
    ]
    assert probe.reached == []


def test_a_server_with_the_denial_extension_answers_the_handshake_with_http() -> None:
    probe = Probe()
    scope = scope_of(
        "websocket", hosts=("evil.example",), extensions={"websocket.http.response": {}}
    )
    start, body = probe.call(scope)
    assert start["type"] == "websocket.http.response.start"
    assert start["status"] == 400
    assert body["type"] == "websocket.http.response.body"
    assert json.loads(body["body"])["error"]["code"] == "host_not_allowed"


def test_a_websocket_that_passes_reaches_the_app() -> None:
    probe = Probe()
    assert probe.call(scope_of("websocket", origins=("http://localhost:8080",))) == []
    assert probe.reached == ["websocket"]


# --- The log ---------------------------------------------------------------------------------


def refusals(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER and record.levelno == logging.WARNING
    ]


@pytest.fixture
def log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        yield caplog


def test_each_refused_host_is_logged_once(log: pytest.LogCaptureFixture) -> None:
    probe = Probe()
    for name in ("evil.example", "evil.example", "other.example", "evil.example"):
        probe.call(scope_of(hosts=(name,)))
    lines = refusals(log)
    assert len(lines) == 2
    assert "evil.example" in lines[0]
    assert "other.example" in lines[1]
    assert all("allowed_hosts" in line for line in lines)


def test_a_refused_origin_is_logged_once_and_apart_from_a_host(
    log: pytest.LogCaptureFixture,
) -> None:
    probe = Probe()
    for _ in range(3):
        probe.call(scope_of("websocket", origins=("http://evil.example",)))
    probe.call(scope_of("websocket", hosts=("http://evil.example",)))
    lines = refusals(log)
    assert len(lines) == 2
    assert "Origin" in lines[0]
    assert "Host" in lines[1]


def test_the_log_never_lists_the_allowed_hosts(log: pytest.LogCaptureFixture) -> None:
    probe = Probe(frozenset({"localhost", "private-name.example", "192.0.2.77"}))
    probe.call(scope_of(hosts=("evil.example",)))
    text = " ".join(refusals(log))
    assert "private-name" not in text
    assert "192.0.2.77" not in text


def test_a_host_with_control_characters_cannot_forge_a_line(log: pytest.LogCaptureFixture) -> None:
    probe = Probe()
    probe.call(scope_of(hosts=("evil\x1b[31m.example\r\nERROR forged line\x00",)))
    (line,) = refusals(log)
    assert "\n" not in line
    assert "\r" not in line
    assert "\x1b" not in line
    assert "\x00" not in line
    assert "\\x1b" in line
    assert "\\x0d\\x0a" in line


def test_a_long_host_is_cut_in_the_log(log: pytest.LogCaptureFixture) -> None:
    probe = Probe()
    probe.call(scope_of(hosts=("a" * 5000,)))
    (line,) = refusals(log)
    assert "a" * 90 in line
    assert "a" * 101 not in line
    assert "..." in line
    assert len(line) < 400


def test_a_non_ascii_host_is_escaped_in_the_log(log: pytest.LogCaptureFixture) -> None:
    probe = Probe()
    probe.call(scope_of(hosts=("b\xfccher.example",)))
    (line,) = refusals(log)
    assert "\\xfc" in line
    assert "\xfc" not in line


def test_the_log_stops_listing_hosts_after_the_cap_and_says_so_once(
    log: pytest.LogCaptureFixture,
) -> None:
    probe = Probe()
    for number in range(MAX_LOGGED_HOSTS + 20):
        probe.call(scope_of(hosts=(f"host{number}.example",)))
    lines = refusals(log)
    assert len(lines) == MAX_LOGGED_HOSTS + 1
    assert "host0.example" in lines[0]
    assert "stops listing" in lines[-1]
    probe.call(scope_of(hosts=("another.example",)))
    assert len(refusals(log)) == MAX_LOGGED_HOSTS + 1  # nothing more, not even a second notice


def test_a_refused_missing_or_repeated_host_is_logged_once(log: pytest.LogCaptureFixture) -> None:
    probe = Probe()
    for _ in range(3):
        probe.call(scope_of(hosts=()))
        probe.call(scope_of(hosts=("a", "b")))
    lines = refusals(log)
    assert len(lines) == 2
    assert "missing" in lines[0]
    assert "repeated 2 times" in lines[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain.example", "plain.example"),
        ("", ""),
        ("a\nb", "a\\x0ab"),
        ("a\tb\x7f", "a\\x09b\\x7f"),
        ("b\xfccher", "b\\xfccher"),
        ("€", "\\u20ac"),
        ("\U0001f600", "\\U0001f600"),
    ],
)
def test_printable_escapes_what_a_log_line_must_not_hold(text: str, expected: str) -> None:
    assert printable(text) == expected


def test_printable_cuts_at_the_limit_with_three_dots() -> None:
    exact = "a" * MAX_LOGGED_CHARS
    assert printable(exact) == exact
    assert printable(exact + "a") == "a" * (MAX_LOGGED_CHARS - 3) + "..."
    assert len(printable("a" * 10_000)) == MAX_LOGGED_CHARS
    assert len(printable("\n" * 10_000)) == MAX_LOGGED_CHARS
    assert printable("abcdef", limit=5) == "ab..."


# --- The settings in the effective configuration ---------------------------------------------


def test_both_host_settings_are_redacted_in_the_effective_configuration_and_the_api(
    tmp_path: Path,
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
) -> None:
    """The real settings model reads the keys, and the redaction hides both of them."""
    local = tmp_path / "local.toml"
    local.write_text(
        '[web]\nbind_address = "192.0.2.9"\nextra_bind_addresses = ["192.0.2.10", "2001:db8::10"]\n'
        'allowed_hosts = ["pi.example", "pi.tailnet.example"]\n',
        encoding="utf-8",
    )
    config = load_config(local_file=local, env={})
    web = config.section("web", WebSettings)
    assert web.listen_addresses() == ("192.0.2.9", "192.0.2.10", "2001:db8::10")
    effective = config.effective(redact=True, omit_site=True)
    assert effective["web"]["allowed_hosts"] == "<redacted>"
    assert effective["web"]["extra_bind_addresses"] == "<redacted>"
    assert effective["web"]["bind_address"] == "<redacted>"
    client = open_client(make_app(config=effective, settings=web), headers={"host": "pi.example"})
    served = client.get(f"{API}/config")
    assert served.status_code == 200
    body = served.json()
    assert body["web"]["allowed_hosts"] == "<redacted>"
    assert body["web"]["extra_bind_addresses"] == "<redacted>"
    for private in ("pi.example", "pi.tailnet", "192.0.2.9", "192.0.2.10", "2001:db8::10"):
        assert private not in served.text
