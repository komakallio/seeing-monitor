"""The dark routes: the library and its session (`GET /dark`), and the command that starts one."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI

from seeingmon.clock import VirtualClock
from seeingmon.scheduler.commands import QueueDark
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import (
    DarkLibraryView,
    DarkModelView,
    DarkStatusView,
    DarkTaskView,
)
from seeingmon.services.web.core_client import (
    CoreProtocolError,
    CoreUnavailableError,
    FakeCoreClient,
)
from seeingmon.services.web.fake_dark import DarkScript
from seeingmon.store.db import Store
from tests.services.web.client import TestClient
from tests.services.web.helpers import bearer, dark_set

API = "/api/v1"
SCRIPT = DarkScript(queued_s=5.0, bias_s=4.0, cover_s=3.0, dark_s=6.0, build_s=1.0)


@pytest.fixture
def dark_core(clock: VirtualClock) -> FakeCoreClient:
    core = FakeCoreClient(clock=clock, dark_script=SCRIPT)
    core.dark.sensor_temperature_c = 12.3
    core.dark.sets = [dark_set("dark-new", 20.0, 10.0), dark_set("dark-old", 4.0, 30.0)]
    core.dark.model = DarkModelView(
        reference_c=20.0, rate_ref_e_per_s=0.12, doubling_c=6.0, doubling_fitted=True, n_sets=2
    )
    return core


@pytest.fixture
def dark_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    dark_core: FakeCoreClient,
) -> TestClient:
    return open_client(make_app(core=dark_core))


def post(client: TestClient, body: Any = None, **kwargs: Any) -> Any:
    return client.post(f"{API}/commands/dark", json=body, headers=bearer(), **kwargs)


# --- The library -----------------------------------------------------------------------------


def test_the_library_has_the_shape_of_the_contract_and_no_note_when_nothing_is_missing(
    dark_client: TestClient, dark_core: FakeCoreClient
) -> None:
    response = dark_client.get(f"{API}/dark")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    expected = dark_core.dark_library().model_dump(mode="json")
    assert response.json() == {**expected, "quality": None}


def test_the_library_carries_the_sets_the_model_the_status_and_the_temperature(
    dark_client: TestClient,
) -> None:
    body = dark_client.get(f"{API}/dark").json()
    assert [item["name"] for item in body["sets"]] == ["dark-new", "dark-old"]
    assert body["sets"][0]["temperature_c"] == 20.0
    assert body["sets"][0]["rate_e_per_s"] == 0.05
    assert body["model"]["doubling_c"] == 6.0
    assert body["sensor_temperature_c"] == 12.3
    assert body["status"]["due"] is True
    assert body["status"]["gap_c"] == 7.7
    assert body["task"]["state"] == "idle"
    assert (body["mode"], body["gain"], body["exposure_s"]) == ("bin2", 120, 30.0)


def test_an_empty_library_says_why_two_values_are_missing(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    empty = FakeCoreClient(clock=clock)
    empty.dark.sensor_temperature_c = None
    body = open_client(make_app(core=empty)).get(f"{API}/dark").json()
    assert body["sets"] == []
    assert body["model"] is None
    assert body["sensor_temperature_c"] is None
    assert body["quality"] == {
        "sensor_temperature_c": "core reports no sensor temperature",
        "model": "core has no dark model yet",
    }
    assert body["status"]["due"] is True


def test_the_progress_of_a_session_shows_in_the_library(
    dark_client: TestClient, clock: VirtualClock
) -> None:
    assert post(dark_client, {"frames": 6, "bias_frames": 4}).status_code == 200
    queued = dark_client.get(f"{API}/dark").json()["task"]
    assert (queued["state"], queued["task_id"], queued["phase"]) == ("queued", 1, None)
    clock.advance(5.0)
    running = dark_client.get(f"{API}/dark").json()["task"]
    assert (running["state"], running["phase"]) == ("running", "bias")
    assert (running["step"], running["steps"]) == (1, 4)
    assert running["message"] == "Bias frame 1 of 4."
    clock.advance(4.0)
    waiting = dark_client.get(f"{API}/dark").json()["task"]
    assert (waiting["phase"], waiting["covered"]) == ("cover", False)
    assert waiting["level_dn"] > 1000
    assert "above the expected level" in waiting["reason"]
    clock.advance(30.0)
    done = dark_client.get(f"{API}/dark").json()
    assert done["task"]["state"] == "ok"
    assert done["task"]["set_name"] == done["sets"][0]["name"]
    assert done["sets"][0]["temperature_c"] == 12.3
    assert done["status"]["due"] is False
    assert dark_client.get(f"{API}/status").json()["scheduler"]["state"] == "paused"


def test_free_text_from_core_loses_its_paths(
    dark_client: TestClient, dark_core: FakeCoreClient
) -> None:
    private = "/home/someone/darks/set.fits"  # repo-check: allow
    view = DarkLibraryView(
        mode="bin2",
        gain=120,
        exposure_s=30.0,
        status=DarkStatusView(
            due=True, reason=f"cannot read {private}", tolerance_c=3.0, max_age_days=183.0
        ),
        task=DarkTaskView(
            state="failed",
            message=f"opened {private}",
            reason=f"wrote {private}",
            summary=f"failed on {private}",
        ),
    )
    dark_core.dark_library = lambda: view  # type: ignore[method-assign]
    response = dark_client.get(f"{API}/dark")
    assert response.status_code == 200
    assert "someone" not in response.text


def test_a_core_that_does_not_answer_is_a_503_and_one_that_answers_nonsense_is_a_502(
    dark_client: TestClient, dark_core: FakeCoreClient
) -> None:
    dark_core.fail_with = CoreUnavailableError("core does not answer")
    response = dark_client.get(f"{API}/dark")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "core_unavailable"
    assert response.headers["retry-after"] == "5"
    dark_core.fail_with = CoreProtocolError("core sent an unreadable dark library")
    assert dark_client.get(f"{API}/dark").status_code == 502


def test_the_library_follows_the_read_rule(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    dark_core: FakeCoreClient,
) -> None:
    locked = WebSettings.model_validate({"require_token_for_reads": True})
    client = open_client(make_app(settings=locked, core=dark_core))
    assert client.get(f"{API}/dark").status_code == 401
    assert client.get(f"{API}/dark", headers=bearer("wrong-token")).status_code == 401
    assert client.get(f"{API}/dark", headers=bearer()).status_code == 200


def test_the_library_needs_no_token_by_default(dark_client: TestClient) -> None:
    assert dark_client.get(f"{API}/dark").status_code == 200


def test_the_library_is_read_only(dark_client: TestClient) -> None:
    assert dark_client.post(f"{API}/dark", json={}, headers=bearer()).status_code == 405


# --- The command -----------------------------------------------------------------------------


def test_an_empty_body_queues_a_session_with_the_defaults(
    dark_client: TestClient, dark_core: FakeCoreClient
) -> None:
    response = post(dark_client, {})
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["task_id"] == 1
    assert body["reason"] is None
    assert "queued" in body["message"]
    assert dark_core.submitted == [QueueDark()]
    assert QueueDark().wait_for_cover is True
    assert QueueDark().pause_after is True


def test_every_field_reaches_core(dark_client: TestClient, dark_core: FakeCoreClient) -> None:
    body = {
        "exposure_s": 45.5,
        "frames": 12,
        "bias_frames": 8,
        "wait_for_cover": False,
        "pause_after": False,
        "label": "after-the-move",
    }
    assert post(dark_client, body).status_code == 200
    assert dark_core.submitted == [
        QueueDark(
            exposure_s=45.5,
            frames=12,
            bias_frames=8,
            wait_for_cover=False,
            pause_after=False,
            label="after-the-move",
        )
    ]


def test_a_null_means_the_configured_value(
    dark_client: TestClient, dark_core: FakeCoreClient
) -> None:
    body = {"exposure_s": None, "frames": None, "bias_frames": None}
    assert post(dark_client, body).status_code == 200
    assert dark_core.submitted == [QueueDark()]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exposure_s", 0.001),
        ("exposure_s", 120),
        ("frames", 3),
        ("frames", 50),
        ("bias_frames", 3),
        ("bias_frames", 50),
        ("label", "a" * 40),
    ],
)
def test_a_value_at_a_bound_is_accepted(
    dark_client: TestClient, dark_core: FakeCoreClient, field: str, value: Any
) -> None:
    assert post(dark_client, {field: value}).status_code == 200
    assert len(dark_core.submitted) == 1


def test_a_second_session_is_a_409_with_the_reason_busy(
    dark_client: TestClient, dark_core: FakeCoreClient
) -> None:
    assert post(dark_client, {}).status_code == 200
    response = post(dark_client, {})
    assert response.status_code == 409
    body = response.json()
    assert (body["accepted"], body["reason"], body["task_id"]) == (False, "busy", None)
    assert "already" in body["message"]


def test_a_session_may_start_again_after_the_first_one_ended(
    dark_client: TestClient, clock: VirtualClock
) -> None:
    assert post(dark_client, {}).status_code == 200
    clock.advance(60)
    assert dark_client.get(f"{API}/dark").json()["task"]["state"] == "ok"
    clock.advance(61)  # the rate limit of the commands forgets the first one
    again = post(dark_client, {})
    assert again.status_code == 200
    assert again.json()["task_id"] == 2


def test_a_pause_cancels_a_running_session_and_the_scheduler_stays_paused(
    dark_client: TestClient, clock: VirtualClock
) -> None:
    post(dark_client, {})
    clock.advance(8)
    assert dark_client.get(f"{API}/dark").json()["task"]["state"] == "running"
    paused = dark_client.post(f"{API}/mode", json={"mode": "paused"}, headers=bearer())
    assert paused.status_code == 200
    body = dark_client.get(f"{API}/dark").json()
    assert body["task"]["state"] == "aborted"
    assert len(body["sets"]) == 2  # nothing was added
    assert dark_client.get(f"{API}/status").json()["scheduler"]["state"] == "paused"


def test_resume_brings_the_scheduler_back_after_a_session(
    dark_client: TestClient, clock: VirtualClock
) -> None:
    post(dark_client, {})
    clock.advance(60)
    assert dark_client.get(f"{API}/status").json()["scheduler"]["state"] == "paused"
    resumed = dark_client.post(f"{API}/mode", json={"mode": "auto"}, headers=bearer())
    assert resumed.status_code == 200
    assert dark_client.get(f"{API}/status").json()["scheduler"]["state"] == "safe"


def test_the_exposure_and_the_label_obey_the_limits_of_the_installation(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    dark_core: FakeCoreClient,
) -> None:
    tight = WebSettings.model_validate(
        {"requests": {"max_dark_exposure_s": 30.0, "max_label_chars": 10}}
    )
    client = open_client(make_app(settings=tight, core=dark_core))
    too_long = post(client, {"exposure_s": 30.5, "label": "x" * 11})
    assert too_long.status_code == 422
    error = too_long.json()["error"]
    assert error["code"] == "invalid_request"
    assert {item["field"] for item in error["details"]} == {"body.exposure_s", "body.label"}
    assert "30" in str(error["details"])
    assert dark_core.submitted == []
    assert post(client, {"exposure_s": 30.0, "label": "x" * 10}).status_code == 200


def test_the_default_limits_are_120_seconds_and_40_characters() -> None:
    limits = WebSettings().requests
    assert limits.max_dark_exposure_s == 120.0
    assert limits.max_label_chars == 40


def test_core_that_rejects_the_values_gives_a_422(
    dark_client: TestClient, dark_core: FakeCoreClient
) -> None:
    from seeingmon.scheduler.commands import CommandResult, RejectReason

    dark_core.submit = lambda command: CommandResult(  # type: ignore[method-assign]
        accepted=False,
        message="frames must be between 3 and 60",
        state="auto",
        reason=RejectReason.INVALID,
    )
    response = post(dark_client, {})
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "frames must be between 3 and 60"


def test_a_core_that_does_not_answer_a_command_is_a_503(
    dark_client: TestClient, dark_core: FakeCoreClient
) -> None:
    dark_core.fail_with = CoreUnavailableError("core does not answer")
    response = post(dark_client, {})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
