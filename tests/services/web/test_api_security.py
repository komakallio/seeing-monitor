"""The security headers, the cache headers, the error handlers, and the token for reads."""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI

from seeingmon.services.web.config import WebSettings
from seeingmon.store.db import Store, StoreReader
from seeingmon.store.layout import DataLayout
from tests.services.web.client import TestClient
from tests.services.web.helpers import TOKEN, bearer, tiny_jpeg
from tests.services.web.seed import write_preview

API = "/api/v1"
STAMP = "20261001T020000.000Z"


@pytest.fixture
def with_image(layout: DataLayout) -> str:
    write_preview(layout, STAMP)
    return f"preview-{STAMP}"


# --- The headers -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/index.html",
        f"{API}/status",
        f"{API}/seeing/latest",
        f"{API}/openapi.json",
        f"{API}/nothing",
        f"{API}/seeing?limit=0",
        "/nothing-here.html",
    ],
)
def test_every_response_carries_the_security_headers(client: TestClient, path: str) -> None:
    headers = client.get(path).headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["cross-origin-opener-policy"] == "same-origin"
    assert headers["cross-origin-resource-policy"] == "same-origin"
    assert "camera=()" in headers["permissions-policy"]
    policy = headers["content-security-policy"]
    for directive in (
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "form-action 'none'",
        "base-uri 'none'",
    ):
        assert directive in policy


def test_the_policy_allows_no_other_origin_for_scripts_styles_or_frames(client: TestClient) -> None:
    policy = client.get("/").headers["content-security-policy"]
    assert "http:" not in policy.replace("ws:", "").replace("wss:", "")
    assert "https:" not in policy
    assert "'unsafe-inline'" not in policy
    assert "'unsafe-eval'" not in policy
    assert "*" not in policy


def test_a_response_to_a_bad_command_carries_the_headers_too(client: TestClient) -> None:
    response = client.post(f"{API}/mode", json={})
    assert response.status_code == 401
    assert response.headers["x-content-type-options"] == "nosniff"


def test_a_body_that_is_too_large_gets_the_headers_too(client: TestClient) -> None:
    response = client.post(f"{API}/commands/replay", json={"padding": "x" * 20_000})
    assert response.status_code == 413
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"


# --- The opener policy needs a secure origin ---------------------------------------------------

OPENER = "cross-origin-opener-policy"
OTHER_SECURITY_HEADERS = (
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
    "cross-origin-resource-policy",
    "permissions-policy",
    "content-security-policy",
)
LISTED_HOSTS = ["foo.localhost", "127.1.2.3", "pi.example", "192.0.2.10", "2001:db8::10"]


@pytest.fixture
def listed_client(
    settings: WebSettings,
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
) -> TestClient:
    """A client of a server that allows `LISTED_HOSTS` and the loopback names."""
    chosen = WebSettings.model_validate({**settings.model_dump(), "allowed_hosts": LISTED_HOSTS})
    return open_client(make_app(settings=chosen))


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "localhost:8080",
        "LOCALHOST.",
        "127.0.0.1",
        "127.0.0.1:8080",
        "[::1]",
        "[::1]:8080",
        "foo.localhost",
        "127.1.2.3",
    ],
)
def test_the_opener_policy_goes_out_on_a_loopback_origin(
    listed_client: TestClient, host: str
) -> None:
    response = listed_client.get(f"{API}/status", headers={"host": host})
    assert response.status_code == 200
    assert response.headers[OPENER] == "same-origin"


@pytest.mark.parametrize(
    "host",
    ["pi.example", "pi.example:8080", "192.0.2.10", "192.0.2.10:8080", "[2001:db8::10]:8080"],
)
def test_the_opener_policy_stays_home_on_a_lan_or_vpn_origin_and_the_other_headers_stay(
    listed_client: TestClient, host: str
) -> None:
    for path in ("/", f"{API}/status", f"{API}/nothing", f"{API}/seeing?limit=0", "/missing.js"):
        response = listed_client.get(path, headers={"host": host})
        assert OPENER not in response.headers, (host, path)
        for name in OTHER_SECURITY_HEADERS:
            assert name in response.headers, (host, path, name)
        assert response.headers["cross-origin-resource-policy"] == "same-origin"
        assert response.headers["x-frame-options"] == "DENY"


def test_the_opener_policy_goes_out_over_https_on_any_host(
    listed_client: TestClient, open_client: Callable[..., TestClient]
) -> None:
    client = open_client(
        listed_client.app, base_url="https://pi.example", headers={"host": "pi.example"}
    )
    response = client.get(f"{API}/status")
    assert response.status_code == 200
    assert response.request.url.scheme == "https"
    assert response.headers[OPENER] == "same-origin"


def test_a_forwarded_scheme_is_not_trusted(listed_client: TestClient) -> None:
    forwarded = {
        "host": "pi.example",
        "x-forwarded-proto": "https",
        "x-forwarded-host": "localhost",
        "forwarded": "proto=https;host=localhost",
    }
    response = listed_client.get(f"{API}/status", headers=forwarded)
    assert response.status_code == 200
    assert OPENER not in response.headers


def test_a_refusal_follows_the_same_rule(client: TestClient) -> None:
    refused = client.get(f"{API}/status", headers={"host": "evil.example"})
    assert refused.status_code == 400
    assert OPENER not in refused.headers
    for name in OTHER_SECURITY_HEADERS:
        assert name in refused.headers
    on_loopback = client.get(f"{API}/status", headers={"host": "localhost"})
    assert on_loopback.headers[OPENER] == "same-origin"


def test_a_response_with_no_usable_host_gets_no_opener_policy_and_all_the_rest() -> None:
    from seeingmon.services.web.middleware import security_headers

    names = {name for name, _ in security_headers({"type": "http", "headers": []})}
    assert OPENER not in names
    assert set(OTHER_SECURITY_HEADERS) <= names
    secure = {
        name for name, _ in security_headers({"type": "http", "scheme": "https", "headers": []})
    }
    assert OPENER in secure


# --- The cache headers -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [f"{API}/status", f"{API}/seeing", f"{API}/events", f"{API}/nothing", f"{API}/openapi.json"],
)
def test_live_data_is_never_stored(client: TestClient, path: str) -> None:
    assert client.get(path).headers["cache-control"] == "no-store"


def test_the_static_files_must_be_checked_again_but_cost_a_304(client: TestClient) -> None:
    first = client.get("/")
    assert first.status_code == 200
    assert first.headers["cache-control"] == "no-cache"
    assert first.headers["content-type"].startswith("text/html")
    etag = first.headers["etag"]
    assert first.headers["last-modified"]
    again = client.get("/", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.content == b""
    assert again.headers["cache-control"] == "no-cache"


def test_an_image_by_id_may_be_cached_for_a_day_and_the_latest_one_may_not(
    client: TestClient, with_image: str
) -> None:
    assert client.get(f"{API}/images/{with_image}").headers["cache-control"] == (
        "public, max-age=86400, immutable"
    )
    assert client.get(f"{API}/images/latest").headers["cache-control"] == "no-cache"


# --- Compression -----------------------------------------------------------------------------


def test_a_large_json_answer_is_compressed_for_a_client_that_accepts_gzip(
    client: TestClient,
) -> None:
    plain = client.get(f"{API}/seeing", headers={"Accept-Encoding": "identity"})
    packed = client.get(f"{API}/seeing", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in plain.headers
    assert packed.headers["content-encoding"] == "gzip"
    assert packed.headers["vary"] == "Accept-Encoding"
    assert packed.json() == plain.json()
    assert len(gzip.compress(plain.content)) < len(plain.content)


def test_a_small_answer_is_not_compressed(client: TestClient) -> None:
    response = client.get(f"{API}/health", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in response.headers


def test_an_image_is_never_compressed(client: TestClient, with_image: str) -> None:
    response = client.get(f"{API}/images/{with_image}", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in response.headers
    assert response.content == tiny_jpeg(50)


# --- The routes that must not exist ----------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["/docs", "/redoc", f"{API}/docs", f"{API}/redoc", "/openapi.json"]
)
def test_the_interactive_docs_that_load_from_a_cdn_are_not_served(
    client: TestClient, path: str
) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert "cdn" not in response.text.lower()
    assert "swagger" not in response.text.lower()


def test_an_unknown_api_path_is_a_404_with_the_error_shape(client: TestClient) -> None:
    response = client.get(f"{API}/nothing/here")
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "The resource does not exist.",
            "details": None,
        }
    }


def test_an_unknown_static_path_is_a_404_with_the_error_shape(client: TestClient) -> None:
    response = client.get("/nothing-here.html")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("method", ["put", "patch", "delete"])
def test_a_method_that_no_route_takes_is_a_405(client: TestClient, method: str) -> None:
    response = getattr(client, method)(f"{API}/status")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


@pytest.mark.parametrize(
    "path",
    [
        "/%2e%2e/app.py",
        "/..%2fapp.py",
        "/%2e%2e%2f%2e%2e%2fconfig.py",
        "/static/../app.py",
        "/..%5capp.py",
        "/%00",
        "/index.html%00.png",
    ],
)
def test_the_static_files_cannot_be_left_by_a_path(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code in {400, 404}
    assert "from __future__" not in response.text
    assert "def create_app" not in response.text


# --- The failures ----------------------------------------------------------------------------


def test_an_unexpected_error_is_a_generic_500_without_a_trace_or_a_detail(
    app: FastAPI,
    seeded: Store,
    open_client: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(record_type: str) -> Any:
        raise RuntimeError("secret detail at the private path")

    monkeypatch.setattr(app.state.ctx.data, "latest", boom)
    client = open_client(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        response = client.get(f"{API}/seeing/latest")
    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "The server hit an unexpected error.",
            "details": None,
        }
    }
    assert "secret detail" not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" in caplog.text
    assert "secret detail" not in caplog.text  # the log line names the type and nothing else


def test_a_store_that_cannot_be_read_is_a_503_with_a_retry_time(
    client: TestClient, reader: StoreReader
) -> None:
    reader.close()
    for path in (f"{API}/seeing/latest", f"{API}/seeing", f"{API}/events"):
        response = client.get(path)
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
        assert response.json()["error"]["code"] == "store_unavailable"


# --- The token for reads ---------------------------------------------------------------------


@pytest.fixture
def locked_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    with_image: str,
) -> TestClient:
    settings = WebSettings.model_validate(
        {"require_token_for_reads": True, "rate_limit": {"auth_failures_per_window": 3}}
    )
    return open_client(make_app(settings=settings))


READS = [
    "/status",
    "/health",
    "/seeing",
    "/seeing/latest",
    "/sky",
    "/sky/latest",
    "/pointing",
    "/pointing/latest",
    "/events",
    "/images",
    "/images/latest",
    "/profile",
    "/config",
    "/alignment/state",
    "/alignment/frame",
]


@pytest.mark.parametrize("path", READS)
def test_with_the_setting_on_a_read_needs_the_token(locked_client: TestClient, path: str) -> None:
    refused = locked_client.get(f"{API}{path}")
    assert refused.status_code == 401
    assert refused.headers["www-authenticate"] == 'Bearer realm="seeing-monitor"'
    assert refused.json()["error"]["code"] == "unauthorized"
    allowed = locked_client.get(f"{API}{path}", headers=bearer())
    assert allowed.status_code in {200, 204, 503}  # the health of the seeded store is fine


def test_the_images_by_id_need_the_token_too(locked_client: TestClient, with_image: str) -> None:
    assert locked_client.get(f"{API}/images/{with_image}").status_code == 401
    assert locked_client.get(f"{API}/images/{with_image}", headers=bearer()).status_code == 200


def test_the_static_ui_stays_open_so_that_the_page_can_ask_for_the_token(
    locked_client: TestClient,
) -> None:
    assert locked_client.get("/").status_code == 200
    assert locked_client.get(f"{API}/openapi.json").status_code == 200


def test_a_wrong_token_on_a_read_counts_as_a_failure_and_ends_in_429(
    locked_client: TestClient,
) -> None:
    for _ in range(3):
        assert locked_client.get(f"{API}/status", headers=bearer("wrong")).status_code == 401
    blocked = locked_client.get(f"{API}/status", headers=bearer())
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "300"


def test_the_token_for_reads_is_the_same_token_as_for_commands(
    locked_client: TestClient,
) -> None:
    assert locked_client.get(f"{API}/status", headers=bearer(TOKEN)).status_code == 200
    assert locked_client.post(f"{API}/alignment/stop", headers=bearer(TOKEN)).status_code == 409


def test_the_status_tells_the_ui_that_reads_need_a_token(locked_client: TestClient) -> None:
    ui = locked_client.get(f"{API}/status", headers=bearer()).json()["ui"]
    assert ui["token_required_for_reads"] is True


def test_the_json_of_a_response_is_valid_json_with_a_utf8_type(client: TestClient) -> None:
    response = client.get(f"{API}/events")
    assert response.headers["content-type"] == "application/json"
    json.loads(response.text)
