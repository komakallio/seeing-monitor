"""The rapid focus mode over the REST and WebSocket routes of the demo app.

The tests drive the demo as a page does: start the alignment, wait for the offer, start the mode
with the demo token, read the state, the video, and the readings, and stop.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.demo import DEMO_TOKEN, DemoApp, build_demo
from seeingmon.services.web.openapi import render_openapi
from tests.records.jsonschema_lite import validate
from tests.services.conftest import wait_until
from tests.services.web.client import TestClient
from tests.services.web.conftest import CONFIG, PROFILE
from tests.services.web.helpers import bearer

API = "/api/v1"
START = f"{API}/alignment/rapid-focus/start"
STOP = f"{API}/alignment/rapid-focus/stop"
STATE = f"{API}/alignment/state"


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DemoApp]:
    settings = WebSettings.model_validate(
        {"rate_limit": {"commands_per_window": 500, "window_s": 60.0}}
    )  # the tests send many commands to one app
    built = build_demo(
        settings,
        profile=PROFILE,
        config=CONFIG,
        directory=tmp_path_factory.mktemp("demo-rapid"),
        frame_period_s=0.05,
        polaris_period_s=0.02,
    )
    yield built
    built.close()


@pytest.fixture
def client(demo: DemoApp, open_client: Callable[..., TestClient]) -> Iterator[TestClient]:
    opened = open_client(demo.app)
    yield opened
    headers = bearer(DEMO_TOKEN)  # leave the demo as the next test finds it
    opened.post(STOP, headers=headers)
    opened.post(f"{API}/alignment/stop", headers=headers)


def rapid(client: TestClient) -> dict[str, Any]:
    view = client.get(STATE).json()["rapid_focus"]
    assert isinstance(view, dict)
    return view


def offered(client: TestClient) -> bool:
    state = client.get(STATE).json()
    view = state.get("rapid_focus")
    return bool(state["active"] and view and view["available"])


def start_the_mode(client: TestClient) -> None:
    headers = bearer(DEMO_TOKEN)
    assert client.post(f"{API}/alignment/start", headers=headers).status_code == 200
    assert wait_until(lambda: offered(client), timeout_s=10.0)
    response = client.post(START, headers=headers)
    assert response.status_code == 200, response.text


class TestTheRoutes:
    def test_the_start_is_refused_outside_the_alignment_and_before_the_offer(
        self, client: TestClient
    ) -> None:
        headers = bearer(DEMO_TOKEN)
        outside = client.post(START, headers=headers)
        assert outside.status_code == 409
        assert outside.json()["reason"] == "not_aligning"
        assert client.post(f"{API}/alignment/start", headers=headers).status_code == 200
        early = client.post(START, headers=headers)
        if early.status_code == 409:  # the first frames have not gone by yet
            assert early.json()["reason"] == "not_available"
            assert "focus values" in early.json()["message"]

    def test_the_offer_comes_after_five_frames_and_the_start_runs_the_mode(
        self, client: TestClient
    ) -> None:
        headers = bearer(DEMO_TOKEN)
        client.post(f"{API}/alignment/start", headers=headers)
        assert wait_until(lambda: offered(client), timeout_s=10.0)
        offer = rapid(client)
        assert (offer["available"], offer["located_by"], offer["active"]) == (
            True,
            "current solution",
            False,
        )
        assert offer["max_fwhm_arcsec"] == 12.0
        started = client.post(START, headers=headers)
        assert started.status_code == 200
        assert started.json()["state"] == "align"
        view = rapid(client)
        assert (view["active"], view["n_stars"]) == (True, 1)
        assert view["readings"]["reset"] is True

    def test_a_start_with_settings_changes_the_stream_and_a_stop_ends_the_mode(
        self, client: TestClient
    ) -> None:
        start_the_mode(client)
        headers = bearer(DEMO_TOKEN)
        changed = client.post(START, json={"exposure_us": 3000, "gain": 20}, headers=headers)
        assert changed.status_code == 200
        view = rapid(client)
        assert (view["exposure_us"], view["gain"]) == (3000, 20)
        stopped = client.post(STOP, headers=headers)
        assert stopped.status_code == 200
        after = rapid(client)
        assert (after["active"], after["ended_reason"]) == (False, "you stopped rapid focus")
        assert after["readings"] is None

    def test_the_status_shows_the_phase_of_the_mode(self, client: TestClient) -> None:
        start_the_mode(client)
        activity = client.get(f"{API}/status").json()["scheduler"]["activity"]
        assert (activity["state"], activity["phase"]) == ("align", "rapid_focus")
        assert activity["label"] == "Rapid focus on Polaris"

    def test_the_state_of_the_mode_matches_the_schema_of_the_api(self, client: TestClient) -> None:
        document = json.loads(render_openapi())
        schema = document["paths"][STATE]["get"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]
        headers = bearer(DEMO_TOKEN)
        client.post(f"{API}/alignment/start", headers=headers)
        validate(client.get(STATE).json(), schema, document)  # the offer
        assert wait_until(lambda: offered(client), timeout_s=10.0)
        assert client.post(START, headers=headers).status_code == 200
        validate(client.get(STATE).json(), schema, document)  # the readings

    def test_the_mode_ends_with_the_alignment(self, client: TestClient) -> None:
        start_the_mode(client)
        assert client.post(f"{API}/alignment/stop", headers=bearer(DEMO_TOKEN)).status_code == 200
        assert client.get(STATE).json()["active"] is False
        client.post(f"{API}/alignment/start", headers=bearer(DEMO_TOKEN))
        assert rapid(client)["ended_reason"] == "the alignment ended"


class TestTheVideo:
    def test_the_websocket_sends_all_the_readings_first_and_then_only_the_new_ones(
        self, client: TestClient
    ) -> None:
        start_the_mode(client)
        seen: list[int] = []
        resets: list[bool] = []
        with client.websocket_connect(f"{API}/polaris/stream") as session:
            for _ in range(8):
                message = session.receive_json()
                assert message["type"] == "state"
                session.receive_bytes()
                readings = message["state"]["rapid_focus"]["readings"]
                resets.append(readings["reset"])
                seen += readings["index"]
        assert resets[0] is True
        assert not any(resets[1:])
        assert seen == list(range(seen[0], seen[0] + len(seen)))  # no gap and no repeat
        assert len(seen) >= 5  # about 20 readings a second, and a message every 50 ms

    def test_the_poll_route_keeps_the_readings_out_of_the_header(self, client: TestClient) -> None:
        start_the_mode(client)

        def served() -> bool:
            # The hub keeps the newest frame that it has, so wait for one of the mode.
            response = client.get(f"{API}/polaris/frame")
            if response.status_code != 200:
                return False
            return json.loads(response.headers["x-frame-state"])["rapid_focus"] is not None

        assert wait_until(served, timeout_s=10.0)
        response = client.get(f"{API}/polaris/frame")
        state = json.loads(response.headers["x-frame-state"])
        assert state["rapid_focus"]["active"] is True
        assert state["rapid_focus"]["readings"] is None
        assert (
            state["quality"]["live_seeing"] == "the rapid focus mode makes no rolling seeing value"
        )
        assert len(response.headers["x-frame-state"]) < 2500
