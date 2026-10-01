"""`/profile` and `/config`, and a sweep that no response carries a secret or the site."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from seeingmon.clock import VirtualClock
from seeingmon.config import load_config
from seeingmon.profile.summary import profile_summary
from seeingmon.scheduler.commands import CommandResult, RejectReason
from seeingmon.services.web.core_client import FakeCoreClient
from seeingmon.store.db import Store
from seeingmon.store.layout import DataLayout
from tests.services.web.client import TestClient
from tests.services.web.helpers import bearer
from tests.services.web.seed import write_fits, write_preview

API = "/api/v1"

# Every value below is made up. A real deployment keeps its own in `local/config.toml`.
LOCAL_CONFIG = """
station_id = "test-station"

[site]
latitude_deg = 12.3456
longitude_deg = 65.4321
elevation_m = 777.7

[paths]
data_dir = "a-private-data-folder"

[replay]
recordings_dir = "a-private-recordings-folder"

[auth]
token_hash = "a-private-token-hash"

[services]
connection_key = "a-private-connection-key-value"
acquire_address = "a-private-acquire-address"

[services.acquire.driver_options]
serial = "PRIVATESERIAL1"  # repo-check: allow
library_path = "a-private-library-path"

[sqm]
enabled = true
host = "sqm-private-host.example.org"

[power]
route = "http"

[power.http]
url = "https://plug-private-host.example.org/cycle"
headers = { Authorization = "Bearer ${PLUG_PRIVATE_TOKEN}" }

[sinks.lab]
kind = "influx"
endpoint = "https://influx-private-host.example.org"
org = "private-org"
bucket = "private-bucket"
token = "a-private-influx-token"

[heater]
enabled = true

[heater.pins.heater]
chip = "private-chip"
line = 17

[heater.ambient]
kind = "sysfs"
temperature_file = "a-private-sensor-file"

[web]
bind_address = "192.0.2.77"
"""

PRIVATE = [
    "12.3456",
    "65.4321",
    "777.7",
    "a-private-data-folder",
    "a-private-recordings-folder",
    "a-private-token-hash",
    "a-private-connection-key-value",
    "a-private-acquire-address",
    "PRIVATESERIAL1",
    "a-private-library-path",
    "sqm-private-host",
    "plug-private-host",
    "PLUG_PRIVATE_TOKEN",
    "influx-private-host",
    "private-org",
    "private-bucket",
    "a-private-influx-token",
    "private-chip",
    "a-private-sensor-file",
    "192.0.2.77",
]


@pytest.fixture
def private_config(tmp_path: Path) -> Any:
    local = tmp_path / "local.toml"
    local.write_text(LOCAL_CONFIG, encoding="utf-8")
    return load_config(local_file=local, env={})


@pytest.fixture
def private_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    private_config: Any,
) -> TestClient:
    app = make_app(
        profile=profile_summary(private_config.profile),
        config=private_config.effective(redact=True, omit_site=True),
        station_id="test-station",
    )
    return open_client(app)


# --- The profile -----------------------------------------------------------------------------


def test_the_profile_is_served_with_its_derived_values(client: TestClient) -> None:
    response = client.get(f"{API}/profile")
    assert response.status_code == 200
    assert response.json()["id"] == "test-profile"
    assert response.json()["readout_modes"][0]["derived"]["plate_scale_arcsec_per_px"] == 1.9


def test_the_real_profile_summary_is_served_as_it_is(private_client: TestClient) -> None:
    body = private_client.get(f"{API}/profile").json()
    assert body["id"] == "asi294mm-gs250"
    modes = {mode["name"]: mode for mode in body["readout_modes"]}
    assert round(modes["bin1"]["derived"]["plate_scale_arcsec_per_px"], 2) == 1.91
    assert round(modes["bin2"]["derived"]["plate_scale_arcsec_per_px"], 2) == 3.82
    assert body["optics"]["derived"]["f_number"] == 5.0


def test_a_server_without_a_profile_says_so(
    make_app: Callable[..., FastAPI], open_client: Callable[..., TestClient], seeded: Store
) -> None:
    client = open_client(make_app(profile=None, config=None))
    for path in ("profile", "config"):
        response = client.get(f"{API}/{path}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


# --- The configuration -----------------------------------------------------------------------


def test_the_configuration_keeps_the_sections_that_describe_the_computation(
    client: TestClient,
) -> None:
    assert client.get(f"{API}/config").json() == {
        "profile": "test-profile",
        "station_id": "test-station",
        "scheduler": {"fast": {"window_s": 120.0}},
    }


def test_the_real_configuration_loses_every_private_value(private_client: TestClient) -> None:
    response = private_client.get(f"{API}/config")
    assert response.status_code == 200
    body = response.json()
    for section in ("site", "paths", "replay", "auth", "sqm", "power", "sinks", "heater"):
        assert section not in body
    assert body["scheduler"]["fast"]["window_s"] == 120.0
    assert body["services"]["acquire"]["driver"] == "sim"
    assert "driver_options" not in body["services"]["acquire"]
    for private in PRIVATE:
        assert private not in response.text


def test_the_configuration_has_no_site_coordinates_even_as_a_nested_value(
    private_client: TestClient,
) -> None:
    text = private_client.get(f"{API}/config").text
    for word in ("latitude", "longitude", "site"):
        assert word not in text.lower().replace("withhold", "")


# --- The sweep -------------------------------------------------------------------------------


GETS = [
    "",
    "/status",
    "/health",
    "/seeing/latest",
    "/seeing",
    "/seeing?step=10m",
    "/sky/latest",
    "/sky",
    "/pointing/latest",
    "/pointing",
    "/events",
    "/images",
    "/images/latest?format=json",
    "/profile",
    "/config",
    "/alignment/state",
    "/openapi.json",
]


def test_no_response_and_no_log_line_carries_a_secret_or_the_site(
    private_client: TestClient,
    layout: DataLayout,
    caplog: pytest.LogCaptureFixture,
    clock: VirtualClock,
) -> None:
    write_preview(layout, "20261001T020000.000Z")
    write_fits(layout, "20261001T020000.000Z")
    caplog.set_level(logging.DEBUG)
    bodies: list[str] = []
    for path in GETS:
        response = private_client.get(f"{API}{path}")
        assert response.status_code == 200, path
        bodies.append(response.text)
        bodies.append(str(response.headers))
    responses = [
        private_client.get(f"{API}/nothing"),
        private_client.get(f"{API}/seeing?limit=0"),
        private_client.post(f"{API}/mode", json={"mode": "auto"}),
        private_client.post(f"{API}/mode", json={"mode": "auto"}, headers=bearer("wrong")),
        private_client.get(f"{API}/images/..%2f..%2fapp"),
    ]
    bodies += [response.text for response in responses]
    everything = "\n".join(bodies) + caplog.text
    for private in PRIVATE:
        assert private not in everything, private


def test_an_error_text_from_core_loses_its_private_values_before_it_reaches_a_client(
    private_client: TestClient,
) -> None:
    class Leaky(FakeCoreClient):
        def submit(self, command: Any) -> CommandResult:
            return CommandResult(
                accepted=False,
                message="failed at /home/someone/x, and 192.0.2.77",  # repo-check: allow
                state="auto",
                reason=RejectReason.PAUSED,
            )

    private_client.app.state.ctx.core = Leaky()  # type: ignore[attr-defined]
    response = private_client.post(f"{API}/mode", json={"mode": "paused"}, headers=bearer())
    assert response.status_code == 409
    assert "someone" not in response.text
    assert "192.0.2.77" not in response.text
