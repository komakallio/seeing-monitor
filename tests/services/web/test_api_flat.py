"""The flat routes: the library and its session, the commands, the decisions, and the preview."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI

from seeingmon.clock import VirtualClock
from seeingmon.scheduler.commands import CancelTask, CommandResult, QueueFlat, RejectReason
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import (
    FlatActionView,
    FlatLibraryView,
    FlatTaskView,
    FlatView,
)
from seeingmon.services.web.core_client import (
    CoreProtocolError,
    CoreUnavailableError,
    FakeCoreClient,
)
from seeingmon.services.web.fake_flat import DARK_FIRST, NO_FIRST_SET, FlatScript
from seeingmon.store.db import Store
from tests.services.web.client import TestClient
from tests.services.web.helpers import bearer, dark_set

API = "/api/v1"
SCRIPT = FlatScript(queued_s=5.0, setup_s=1.0, exposure_s=2.0, capture_s=8.0, build_s=1.0)
UNKNOWN = "flat-00000000"


@pytest.fixture
def flat_core(clock: VirtualClock) -> FakeCoreClient:
    core = FakeCoreClient(clock=clock, flat_script=SCRIPT)
    core.dark.sets = [dark_set("dark-a", 12.0, 10.0)]
    core.flat.sensor_temperature_c = 12.3
    core.flat.seed(age_days=30.0, active=True)
    core.flat.seed(age_days=1.0, state="pending")
    return core


@pytest.fixture
def flat_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    flat_core: FakeCoreClient,
) -> TestClient:
    return open_client(make_app(core=flat_core))


def session(client: TestClient, body: Any = None, **kwargs: Any) -> Any:
    return client.post(f"{API}/flat/session", json=body, headers=bearer(), **kwargs)


def newest(core: FakeCoreClient) -> FlatView:
    return core.flat_library().flats[0]


def active(core: FakeCoreClient) -> FlatView:
    return next(item for item in core.flat_library().flats if item.active)


# --- The library -----------------------------------------------------------------------------


def test_the_library_has_the_shape_of_the_contract_plus_the_address_of_each_image(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    response = flat_client.get(f"{API}/flat")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    expected = flat_core.flat_library().model_dump(mode="json")
    for item in expected["flats"]:
        item["image_url"] = f"{API}/flat/{item['version']}/image"
    assert response.json() == {**expected, "quality": None}


def test_the_library_lists_the_newest_flat_first_and_names_the_active_and_the_pending_one(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    body = flat_client.get(f"{API}/flat").json()
    assert [item["state"] for item in body["flats"]] == ["pending", "approved"]
    assert body["pending_version"] == body["flats"][0]["version"]
    assert body["active_version"] == body["flats"][1]["version"]
    assert [item["active"] for item in body["flats"]] == [False, True]
    assert body["flats"][0]["corner_percent"] == -9.6
    assert body["flats"][0]["warnings"]  # one set cannot tell the source from the optics
    assert (body["mode"], body["gain"], body["blocker"]) == ("bin2", 120, None)
    assert body["sensor_temperature_c"] == 12.3
    assert body["task"]["state"] == "idle"
    assert body["session"] is None


def test_an_empty_library_says_why_the_temperature_is_missing_and_why_no_session_can_start(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    empty = FakeCoreClient(clock=clock)
    empty.flat.sensor_temperature_c = None
    body = open_client(make_app(core=empty)).get(f"{API}/flat").json()
    assert body["flats"] == []
    assert body["sensor_temperature_c"] is None
    assert body["quality"] == {"sensor_temperature_c": "core reports no sensor temperature"}
    assert body["blocker"] == DARK_FIRST  # no dark set


def test_the_library_says_that_it_wins_over_a_pinned_flat_file(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    flat_core.flat.flat_file_pinned = True
    body = flat_client.get(f"{API}/flat").json()
    assert (body["flat_file_pinned"], body["library_overrides"]) == (True, True)


def test_the_progress_of_a_session_shows_in_the_library(
    flat_client: TestClient, clock: VirtualClock
) -> None:
    assert session(flat_client, {"frames": 16}).status_code == 200
    queued = flat_client.get(f"{API}/flat").json()["task"]
    assert (queued["state"], queued["task_id"], queued["phase"]) == ("queued", 1, None)
    clock.advance(5.0)
    running = flat_client.get(f"{API}/flat").json()["task"]
    assert (running["state"], running["phase"], running["message"]) == (
        "running",
        "setup",
        "Setting up the camera and the library.",
    )
    clock.advance(1.2)  # the search for the exposure has begun, and no try has ended
    start = flat_client.get(f"{API}/flat").json()["task"]
    assert (start["phase"], start["step"], start["steps"]) == ("exposure", 0, 8)
    assert start["message"] == "Finding the exposure that reaches 50 % of full scale."
    assert start["level_fraction"] is None
    clock.advance(1.0)  # the first try, which misses the target
    first = flat_client.get(f"{API}/flat").json()["task"]
    assert (first["phase"], first["step"]) == ("exposure", 1)
    assert first["level_fraction"] == pytest.approx(0.256, abs=0.01)
    clock.advance(0.6)  # the second try
    search = flat_client.get(f"{API}/flat").json()["task"]
    assert (search["phase"], search["step"], search["steps"]) == ("exposure", 2, 8)
    assert search["level_fraction"] == pytest.approx(0.5, abs=0.01)
    clock.advance(1.0)  # the first frames
    capture = flat_client.get(f"{API}/flat").json()["task"]
    assert (capture["phase"], capture["steps"], capture["frames"]) == ("capture", 16, 16)
    assert capture["target_fraction"] == 0.5
    clock.advance(30.0)
    done = flat_client.get(f"{API}/flat").json()
    assert done["task"]["state"] == "ok"
    assert done["task"]["version"] == done["flats"][0]["version"]
    assert done["pending_version"] == done["task"]["version"]
    assert done["session"]["version"] == done["task"]["version"]
    assert flat_client.get(f"{API}/status").json()["scheduler"]["state"] == "paused"


def test_free_text_from_core_loses_its_paths(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    private = "/home/someone/flats/set1.ser"  # repo-check: allow
    library = flat_core.flat_library()
    flat = library.flats[0].model_copy(
        update={"warnings": [f"read {private}"], "bias_note": f"from {private}"}
    )
    view = library.model_copy(
        update={
            "flats": [flat],
            "blocker": f"cannot open {private}",
            "task": FlatTaskView(
                state="failed",
                message=f"opened {private}",
                summary=f"failed on {private}",
                warnings=[f"wrote {private}"],
            ),
        }
    )
    flat_core.flat_library = lambda: view  # type: ignore[method-assign]
    response = flat_client.get(f"{API}/flat")
    assert response.status_code == 200
    assert "someone" not in response.text
    assert "<hidden>" in response.text


def test_a_core_that_does_not_answer_is_a_503_and_one_that_answers_nonsense_is_a_502(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    flat_core.fail_with = CoreUnavailableError("core does not answer")
    response = flat_client.get(f"{API}/flat")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "core_unavailable"
    assert response.headers["retry-after"] == "5"
    flat_core.fail_with = CoreProtocolError("core sent an unreadable flat library")
    assert flat_client.get(f"{API}/flat").status_code == 502


def test_the_library_follows_the_read_rule(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    flat_core: FakeCoreClient,
) -> None:
    locked = WebSettings.model_validate({"require_token_for_reads": True})
    client = open_client(make_app(settings=locked, core=flat_core))
    version = newest(flat_core).version
    for path in ("/flat", f"/flat/{version}/image"):
        assert client.get(f"{API}{path}").status_code == 401
        assert client.get(f"{API}{path}", headers=bearer("wrong-token")).status_code == 401
        assert client.get(f"{API}{path}", headers=bearer()).status_code == 200


def test_the_library_needs_no_token_by_default_and_is_read_only(flat_client: TestClient) -> None:
    assert flat_client.get(f"{API}/flat").status_code == 200
    assert flat_client.put(f"{API}/flat", json={}, headers=bearer()).status_code == 405
    assert flat_client.post(f"{API}/flat", json={}, headers=bearer()).status_code == 405


# --- The session -----------------------------------------------------------------------------


def test_an_empty_body_queues_a_session_with_the_defaults(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    response = session(flat_client, {})
    assert response.status_code == 200
    body = response.json()
    assert (body["accepted"], body["task_id"], body["reason"]) == (True, 1, None)
    assert "queued" in body["message"]
    assert flat_core.submitted == [QueueFlat()]
    assert QueueFlat().frames == 32
    assert QueueFlat().target_fraction == 0.5
    assert QueueFlat().pause_after is True
    assert QueueFlat().immediate is True  # someone holds the light source at the camera


def test_a_request_without_a_body_is_a_422_because_the_body_is_required(
    flat_client: TestClient,
) -> None:
    response = flat_client.post(f"{API}/flat/session", headers=bearer())
    assert response.status_code == 422


def test_every_field_reaches_core(flat_client: TestClient, flat_core: FakeCoreClient) -> None:
    body = {
        "frames": 24,
        "target_fraction": 0.42,
        "set_number": 1,
        "pause_after": False,
        "immediate": False,
        "priority": -3,
    }
    assert session(flat_client, body).status_code == 200
    assert flat_core.submitted == [
        QueueFlat(
            frames=24,
            target_fraction=0.42,
            set_number=1,
            pause_after=False,
            priority=-3,
            immediate=False,
        )
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frames", 8),
        ("frames", 64),
        ("target_fraction", 0.3),
        ("target_fraction", 0.7),
        ("pause_after", False),
        ("immediate", False),
        ("priority", 10),
        ("priority", -10),
    ],
)
def test_a_value_at_a_bound_is_accepted(
    flat_client: TestClient, flat_core: FakeCoreClient, field: str, value: Any
) -> None:
    assert session(flat_client, {field: value}).status_code == 200
    assert len(flat_core.submitted) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"frames": 7},
        {"frames": 65},
        {"frames": 12.5},
        {"frames": "16"},
        {"frames": True},
        {"frames": None},
        {"target_fraction": 0.29},
        {"target_fraction": 0.71},
        {"target_fraction": "0.5"},
        {"target_fraction": float("nan")},
        {"target_fraction": None},
        {"set_number": 0},
        {"set_number": 3},
        {"set_number": "2"},
        {"pause_after": "no"},
        {"pause_after": 0},
        {"pause_after": None},
        {"immediate": "yes"},
        {"priority": 11},
        {"priority": -11},
        {"priority": 1.5},
        {"label": "no label field"},
        {"unknown": 1},
    ],
    ids=lambda body: str(body)[:40],
)
def test_a_value_out_of_range_or_of_the_wrong_type_is_a_422_and_never_reaches_core(
    flat_client: TestClient, flat_core: FakeCoreClient, body: dict[str, Any]
) -> None:
    response = flat_client.post(
        f"{API}/flat/session",
        content=json.dumps(body, allow_nan=True).encode(),  # a client may send NaN
        headers={**bearer(), "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"]
    assert flat_core.submitted == []


def test_a_second_session_is_a_409_with_the_reason_busy(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    assert session(flat_client, {}).status_code == 200
    response = session(flat_client, {})
    assert response.status_code == 409
    body = response.json()
    assert (body["accepted"], body["reason"], body["task_id"]) == (False, "busy", None)
    assert "already" in body["message"]


def test_a_session_without_a_dark_set_is_a_422_that_names_the_dark_page(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    core = FakeCoreClient(clock=clock, flat_script=SCRIPT)
    response = session(open_client(make_app(core=core)), {})
    assert response.status_code == 422
    assert response.json()["error"]["message"] == DARK_FIRST
    assert core.flat.task().state == "idle"


def test_a_second_set_needs_the_first_set_of_a_session(flat_client: TestClient) -> None:
    refused = session(flat_client, {"set_number": 2})
    assert refused.status_code == 422
    assert refused.json()["error"]["message"] == NO_FIRST_SET


def test_a_second_set_makes_one_flat_from_both(
    flat_client: TestClient, flat_core: FakeCoreClient, clock: VirtualClock
) -> None:
    assert session(flat_client, {"frames": 16}).status_code == 200
    clock.advance(30.0)
    first = flat_client.get(f"{API}/flat").json()
    assert first["session"] is not None
    clock.advance(61.0)  # the rate limit of the commands forgets the first one
    resume = flat_client.post(f"{API}/mode", json={"mode": "auto"}, headers=bearer())
    assert resume.status_code == 200
    assert session(flat_client, {"frames": 16, "set_number": 2}).status_code == 200
    clock.advance(30.0)
    second = flat_client.get(f"{API}/flat").json()
    assert second["session"] is None
    assert second["task"]["set_number"] == 2
    assert second["task"]["state"] == "ok"
    assert [item["version"] for item in second["flats"]].count(first["session"]["version"]) == 0
    two_sets = second["flats"][0]
    assert (two_sets["second_set"], two_sets["source_turned"]) == (True, True)
    assert two_sets["optics_tilt"] is not None
    assert two_sets["source_tilt"] is not None
    assert two_sets["agreement"]["plane"] is not None


def test_core_that_rejects_the_values_gives_a_422(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    flat_core.submit = lambda command: CommandResult(  # type: ignore[method-assign]
        accepted=False,
        message="frames must be between 8 and 64",
        state="auto",
        reason=RejectReason.INVALID,
    )
    response = session(flat_client, {})
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "frames must be between 8 and 64"


def test_a_core_that_does_not_answer_a_command_is_a_503(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    flat_core.fail_with = CoreUnavailableError("core does not answer")
    response = session(flat_client, {})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"


def test_a_pause_stops_a_running_session_and_the_scheduler_stays_paused(
    flat_client: TestClient, clock: VirtualClock
) -> None:
    session(flat_client, {})
    clock.advance(9)
    assert flat_client.get(f"{API}/flat").json()["task"]["state"] == "running"
    paused = flat_client.post(f"{API}/mode", json={"mode": "paused"}, headers=bearer())
    assert paused.status_code == 200
    body = flat_client.get(f"{API}/flat").json()
    assert body["task"]["state"] == "aborted"
    assert len(body["flats"]) == 2  # nothing was added
    assert flat_client.get(f"{API}/status").json()["scheduler"]["state"] == "paused"


# --- The stop --------------------------------------------------------------------------------


def stop(client: TestClient, **kwargs: Any) -> Any:
    return client.post(f"{API}/flat/session/stop", headers=bearer(), **kwargs)


def test_stopping_a_session_that_waits_removes_it(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    session(flat_client, {})
    response = stop(flat_client)
    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert flat_core.submitted[-1] == CancelTask(kind="flat")
    task = flat_client.get(f"{API}/flat").json()["task"]
    assert (task["state"], task["summary"]) == (
        "aborted",
        "The flat session was cancelled before it started.",
    )


def test_stopping_a_running_session_ends_it_and_the_scheduler_pauses(
    flat_client: TestClient, clock: VirtualClock
) -> None:
    session(flat_client, {})
    clock.advance(9)
    response = stop(flat_client)
    assert response.status_code == 200
    assert "stops at its next check" in response.json()["message"]
    body = flat_client.get(f"{API}/flat").json()
    assert body["task"]["state"] == "aborted"
    assert len(body["flats"]) == 2
    assert flat_client.get(f"{API}/status").json()["scheduler"]["state"] == "paused"


def test_stopping_without_a_session_is_a_409_with_the_reason_no_task(
    flat_client: TestClient,
) -> None:
    response = stop(flat_client)
    assert response.status_code == 409
    body = response.json()
    assert (body["accepted"], body["reason"]) == (False, "no_task")
    assert "no flat task" in body["message"]


# --- The decisions ---------------------------------------------------------------------------


def activate(client: TestClient, version: str, **kwargs: Any) -> Any:
    return client.post(f"{API}/flat/{version}/activate", headers=bearer(), **kwargs)


def delete(client: TestClient, version: str, **kwargs: Any) -> Any:
    return client.delete(f"{API}/flat/{version}", headers=bearer(), **kwargs)


def test_activating_a_flat_makes_it_the_one_in_use(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    pending = newest(flat_core).version
    response = activate(flat_client, pending)
    assert response.status_code == 200
    assert response.json() == {
        "message": f"The flat {pending} is in use. The survey divides by it from its next frame.",
        "version": pending,
    }
    body = flat_client.get(f"{API}/flat").json()
    assert (body["active_version"], body["pending_version"]) == (pending, None)
    assert body["flats"][0]["state"] == "approved"


def test_an_unknown_flat_is_a_404_for_every_decision_and_for_the_image(
    flat_client: TestClient, clock: VirtualClock
) -> None:
    for call in (activate, delete):
        response = call(flat_client, UNKNOWN)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "unknown_flat"
        clock.advance(61)
    image = flat_client.get(f"{API}/flat/{UNKNOWN}/image")
    assert image.status_code == 404
    assert image.json()["error"]["code"] == "unknown_flat"


@pytest.mark.parametrize(
    "version",
    [
        "flat-1234567",
        "flat-123456789",
        "FLAT-12345678",
        "flat-1234567g",
        "session",
    ],
)
def test_a_name_that_is_no_version_is_a_422_before_it_reaches_core(
    flat_client: TestClient,
    flat_core: FakeCoreClient,
    version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def spy(name: str) -> None:
        calls.append(name)

    for method in ("flat_activate", "flat_delete", "flat_image"):
        monkeypatch.setattr(flat_core, method, spy)
    assert activate(flat_client, version).status_code == 422
    assert delete(flat_client, version).status_code == 422
    assert flat_client.get(f"{API}/flat/{version}/image").status_code == 422
    assert calls == []


def test_the_flat_in_use_cannot_be_deleted(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    in_use = active(flat_core).version
    response = delete(flat_client, in_use)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "flat_in_use"
    assert "in use" in response.json()["error"]["message"]
    assert len(flat_client.get(f"{API}/flat").json()["flats"]) == 2


def test_a_flat_that_is_not_in_use_can_be_deleted(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    pending = newest(flat_core).version
    response = delete(flat_client, pending)
    assert response.status_code == 200
    assert response.json() == {"message": f"The flat {pending} is deleted.", "version": pending}
    assert [item["version"] for item in flat_client.get(f"{API}/flat").json()["flats"]] == [
        active(flat_core).version
    ]


def test_no_decision_can_come_while_a_session_waits_or_runs(
    flat_client: TestClient, flat_core: FakeCoreClient, clock: VirtualClock
) -> None:
    pending = newest(flat_core).version
    session(flat_client, {})
    for call in (activate, delete):
        response = call(flat_client, pending)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "session_busy"
    clock.advance(61)
    stop(flat_client)
    assert activate(flat_client, pending).status_code == 200


def test_an_unusable_flat_is_a_409_with_its_own_code(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    flat_core.flat_activate = lambda version: FlatActionView(  # type: ignore[method-assign]
        ok=False, reason="invalid", message="The flat has 640 x 480 pixels.", version=version
    )
    response = activate(flat_client, newest(flat_core).version)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "flat_unusable"


def test_the_message_of_a_decision_loses_its_paths(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    private = "/home/someone/flats/flat.npy"  # repo-check: allow
    flat_core.flat_activate = lambda version: FlatActionView(  # type: ignore[method-assign]
        ok=True, message=f"Wrote {private}", version=version
    )
    response = activate(flat_client, newest(flat_core).version)
    assert response.status_code == 200
    assert "someone" not in response.text


def test_a_core_that_does_not_answer_a_decision_is_a_503(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    flat_core.fail_with = CoreUnavailableError("core does not answer")
    assert activate(flat_client, UNKNOWN).status_code == 503


# --- The token -------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["post", "delete"])
def test_a_decision_needs_the_token(
    flat_client: TestClient, flat_core: FakeCoreClient, method: str
) -> None:
    version = newest(flat_core).version
    path = f"{API}/flat/{version}" + ("/activate" if method == "post" else "")
    before = flat_core.flat_library()
    for headers in ({}, bearer("wrong-token")):
        response = getattr(flat_client, method)(path, headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == 'Bearer realm="seeing-monitor"'
    assert flat_core.flat_library().flats == before.flats  # nothing changed


def test_a_server_without_a_token_hash_refuses_every_flat_command(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    flat_core: FakeCoreClient,
    clock: VirtualClock,
) -> None:
    client = open_client(make_app(token_hash=None, core=flat_core))
    version = newest(flat_core).version
    calls: list[Callable[[], Any]] = [
        lambda: client.post(f"{API}/flat/session", json={}, headers=bearer()),
        lambda: client.post(f"{API}/flat/session/stop", headers=bearer()),
        lambda: client.post(f"{API}/flat/{version}/activate", headers=bearer()),
        lambda: client.delete(f"{API}/flat/{version}", headers=bearer()),
    ]
    for call in calls:
        response = call()
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "commands_disabled"
        clock.advance(61)
    assert flat_core.submitted == []
    assert len(flat_core.flat_library().flats) == 2


def test_a_delete_counts_against_the_limit_of_commands(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    for _ in range(5):  # the limit of the test settings is 5 for each 60 s
        assert delete(flat_client, UNKNOWN).status_code == 404
    response = delete(flat_client, UNKNOWN)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"


# --- The preview -----------------------------------------------------------------------------


def test_the_image_is_a_jpeg_that_a_client_may_keep(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    version = newest(flat_core).version
    response = flat_client.get(f"{API}/flat/{version}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=86400, immutable"
    assert response.content.startswith(b"\xff\xd8\xff")
    assert response.content == flat_core.flat_image(version)


def test_the_address_of_an_image_in_the_library_works(flat_client: TestClient) -> None:
    for item in flat_client.get(f"{API}/flat").json()["flats"]:
        assert item["has_image"] is True
        assert flat_client.get(item["image_url"]).status_code == 200


def test_a_flat_without_a_preview_has_no_address(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    library: FlatLibraryView = flat_core.flat_library()
    bare = library.model_copy(
        update={"flats": [item.model_copy(update={"has_image": False}) for item in library.flats]}
    )
    flat_core.flat_library = lambda: bare  # type: ignore[method-assign]
    body = flat_client.get(f"{API}/flat").json()
    assert [(item["has_image"], item["image_url"]) for item in body["flats"]] == [
        (False, None),
        (False, None),
    ]


def test_a_core_that_does_not_answer_the_image_is_a_503(
    flat_client: TestClient, flat_core: FakeCoreClient
) -> None:
    version = newest(flat_core).version
    flat_core.fail_with = CoreUnavailableError("core does not answer")
    assert flat_client.get(f"{API}/flat/{version}/image").status_code == 503
