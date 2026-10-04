"""The `POST` routes: the token, the rate limits, the validation, and the scheduler commands."""

from __future__ import annotations

import hmac
import logging
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI

from seeingmon.clock import VirtualClock
from seeingmon.frames import PixelFormat, Roi, StreamConfig
from seeingmon.scheduler.commands import (
    CommandResult,
    Pause,
    QueueBurst,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.core_client import (
    CoreProtocolError,
    CoreUnavailableError,
    FakeCoreClient,
)
from seeingmon.store.db import Store
from tests.services.web.client import TestClient
from tests.services.web.helpers import TOKEN, bearer

API = "/api/v1"

COMMANDS: list[tuple[str, dict[str, Any]]] = [
    ("/commands/burst", {}),
    ("/commands/sweep", {}),
    ("/commands/replay", {"source": "night-1"}),
    ("/commands/dark", {}),
    ("/flat/session", {}),
    ("/flat/session/stop", {}),
    ("/flat/flat-1a2b3c4d/activate", {}),
    ("/mode", {"mode": "paused"}),
    ("/alignment/start", {}),
    ("/alignment/stop", {}),
]


def post(client: TestClient, path: str, body: Any = None, **kwargs: Any) -> Any:
    return client.post(f"{API}{path}", json=body, **kwargs)


# --- The token -------------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "body"), COMMANDS)
def test_a_command_without_a_token_is_refused_before_anything_reaches_core(
    client: TestClient, core: FakeCoreClient, path: str, body: dict[str, Any]
) -> None:
    response = post(client, path, body)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer realm="seeing-monitor"'
    assert response.json() == {
        "error": {
            "code": "unauthorized",
            "message": "The request needs a valid token.",
            "details": None,
        }
    }
    assert core.submitted == []


WRONG_HEADERS = [
    {},
    {"Authorization": "Bearer wrong-token"},
    {"Authorization": f"Bearer {TOKEN}x"},
    {"Authorization": f"Bearer {TOKEN[:-1]}"},
    {"Authorization": f"bearer {TOKEN.upper()}"},
    {"Authorization": f"Basic {TOKEN}"},
    {"Authorization": TOKEN},
    {"Authorization": "Bearer"},
    {"Authorization": "Bearer "},
    {"Authorization": "Bearer a b"},
    {"Authorization": "Bearer " + "x" * 1000},
    {"Authorization": b"Bearer \xe9\xe9"},
    {"Authorization": f"Bearer {TOKEN}, Bearer {TOKEN}"},
    {"authorization": "", "X-Token": TOKEN},
]


def test_wrong_missing_and_malformed_tokens_get_one_and_the_same_answer(
    client: TestClient, core: FakeCoreClient, clock: VirtualClock
) -> None:
    answers = set()
    for headers in WRONG_HEADERS:
        response = post(client, "/mode", {"mode": "paused"}, headers=headers)
        assert response.status_code == 401, headers
        answers.add((response.text, response.headers["www-authenticate"]))
        clock.advance(400)  # both limits forget the client, so the next try starts fresh
    assert len(answers) == 1
    assert core.submitted == []


def test_the_right_token_is_accepted_whatever_the_case_of_the_scheme(
    client: TestClient, core: FakeCoreClient
) -> None:
    for scheme in ("Bearer", "bearer", "BEARER"):
        headers = {"Authorization": f"{scheme} {TOKEN}"}
        response = post(client, "/alignment/start", None, headers=headers)
        assert response.status_code == 200
    assert len(core.submitted) == 3


def test_every_wrong_token_costs_one_scrypt_run_and_is_compared_in_constant_time(
    client: TestClient, clock: VirtualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((len(a), len(b)))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    for token in ("a", "a-much-longer-wrong-token-than-the-first-one", "x" * 200):
        assert post(client, "/mode", {"mode": "auto"}, headers=bearer(token)).status_code == 401
        clock.advance(400)
    assert calls
    assert set(calls) == {(32, 32)}


def test_the_token_never_appears_in_a_response_or_a_log_line(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    wrong = "wrong-token-that-must-not-be-logged-0000"
    for headers in (bearer(wrong), bearer(TOKEN), {}):
        response = post(client, "/mode", {"mode": "paused"}, headers=headers)
        assert wrong not in response.text
        assert TOKEN not in response.text
        assert TOKEN not in str(response.headers)
    assert wrong not in caplog.text
    assert TOKEN not in caplog.text
    assert any("does not match" in record.getMessage() for record in caplog.records)


def test_a_server_without_a_token_hash_refuses_every_command(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    core: FakeCoreClient,
    clock: VirtualClock,
) -> None:
    client = open_client(make_app(token_hash=None))
    for path, body in COMMANDS:
        for headers in ({}, bearer()):
            response = post(client, path, body, headers=headers)
            assert response.status_code == 403
            assert response.json()["error"]["code"] == "commands_disabled"
            clock.advance(61)  # a new window of the command limit
    assert core.submitted == []


def test_a_malformed_token_hash_stops_the_server_at_start(
    make_app: Callable[..., FastAPI],
) -> None:
    from seeingmon.services.web.auth import AuthConfigError

    with pytest.raises(AuthConfigError):
        make_app(token_hash="<hash of the API token>")


def test_reads_need_no_token_by_default(client: TestClient) -> None:
    assert client.get(f"{API}/status").status_code == 200
    assert client.get(f"{API}/alignment/state").status_code == 200


# --- The rate limits -------------------------------------------------------------------------


def test_a_client_that_sends_too_many_commands_gets_429_with_a_retry_time(
    client: TestClient, clock: VirtualClock, core: FakeCoreClient
) -> None:
    for _ in range(5):  # the limit of the test settings is 5 for each 60 s
        assert post(client, "/alignment/start", None, headers=bearer()).status_code == 200
    response = post(client, "/alignment/start", None, headers=bearer())
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert response.json()["error"]["code"] == "rate_limited"
    assert len(core.submitted) == 5
    clock.advance(30)
    assert post(client, "/alignment/start", None, headers=bearer()).status_code == 429
    assert post(client, "/alignment/start", None, headers=bearer()).headers["retry-after"] == "30"
    clock.advance(30)
    assert post(client, "/alignment/start", None, headers=bearer()).status_code == 200


def test_the_command_limit_counts_a_wrong_token_too(client: TestClient) -> None:
    for _ in range(5):
        post(client, "/mode", {"mode": "auto"}, headers=bearer("wrong"))
    assert post(client, "/mode", {"mode": "auto"}, headers=bearer()).status_code == 429


def test_a_client_that_fails_the_token_check_too_often_is_blocked_even_with_the_right_token(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
    core: FakeCoreClient,
) -> None:
    settings = WebSettings.model_validate(
        {"rate_limit": {"commands_per_window": 100, "auth_failures_per_window": 3}}
    )
    client = open_client(make_app(settings=settings))
    for _ in range(3):
        assert post(client, "/mode", {"mode": "auto"}, headers=bearer("wrong")).status_code == 401
    blocked = post(client, "/mode", {"mode": "auto"}, headers=bearer())
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == "300"
    assert core.submitted == []
    clock.advance(301)
    after = post(client, "/mode", {"mode": "auto"}, headers=bearer())
    assert after.status_code == 409  # the check passes, and the scheduler is not paused


def test_a_missing_token_does_not_count_as_a_failed_guess(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
) -> None:
    settings = WebSettings.model_validate(
        {"rate_limit": {"commands_per_window": 100, "auth_failures_per_window": 2}}
    )
    client = open_client(make_app(settings=settings))
    for _ in range(10):
        assert post(client, "/mode", {"mode": "auto"}).status_code == 401
    assert post(client, "/mode", {"mode": "auto"}, headers=bearer()).status_code == 409


def test_reads_are_not_rate_limited_by_the_command_limit(client: TestClient) -> None:
    for _ in range(20):
        assert client.get(f"{API}/status").status_code == 200


# --- The commands that reach core ------------------------------------------------------------


def test_the_burst_route_builds_a_queue_burst(client: TestClient, core: FakeCoreClient) -> None:
    body = {
        "duration_s": 30,
        "label": "focus run",
        "priority": 2,
        "stream": {
            "mode": "bin1",
            "exposure_us": 2000,
            "gain": 10,
            "pixel_format": "RAW8",
            "roi": {"x": 8, "y": 8, "width": 128, "height": 128},
            "high_speed": True,
        },
    }
    response = post(client, "/commands/burst", body, headers=bearer())
    assert response.status_code == 200
    assert response.json() == {
        "accepted": True,
        "message": "the burst is queued and runs at the next cycle boundary",
        "state": "auto",
        "reason": None,
        "task_id": 1,
    }
    assert core.submitted == [
        QueueBurst(
            duration_s=30.0,
            stream=StreamConfig(
                mode="bin1",
                exposure_us=2000,
                gain=10,
                pixel_format=PixelFormat.RAW8,
                roi=Roi(8, 8, 128, 128),
                high_speed=True,
            ),
            label="focus run",
            priority=2,
        )
    ]


def test_the_burst_defaults_to_the_fast_stream_for_ten_seconds(
    client: TestClient, core: FakeCoreClient
) -> None:
    assert post(client, "/commands/burst", {}, headers=bearer()).status_code == 200
    assert core.submitted == [QueueBurst()]


def test_the_sweep_route_builds_a_queue_sweep(client: TestClient, core: FakeCoreClient) -> None:
    body = {
        "exposure_us": [500, 1000],
        "gain": [0, 60],
        "roi_arcmin": [4.1],
        "modes": ["bin1"],
        "window_s": 5,
        "priority": -1,
    }
    assert post(client, "/commands/sweep", body, headers=bearer()).status_code == 200
    assert core.submitted == [
        QueueSweep(
            exposure_us=(500, 1000),
            gain=(0, 60),
            roi_arcmin=(4.1,),
            modes=("bin1",),
            window_s=5.0,
            priority=-1,
        )
    ]


def test_the_replay_route_builds_a_queue_replay(client: TestClient, core: FakeCoreClient) -> None:
    body = {
        "source": "night-1.ser",
        "speed": 0,
        "options": {"loop": True, "start_s": 12.5, "note": "x"},
        "priority": 1,
    }
    assert post(client, "/commands/replay", body, headers=bearer()).status_code == 200
    assert core.submitted == [
        QueueReplay(
            source="night-1.ser",
            speed=0.0,
            options={"loop": True, "start_s": 12.5, "note": "x"},
            priority=1,
        )
    ]


def test_the_mode_route_pauses_and_resumes(client: TestClient, core: FakeCoreClient) -> None:
    paused = post(client, "/mode", {"mode": "paused"}, headers=bearer())
    assert paused.status_code == 200
    assert paused.json()["state"] == "paused"
    resumed = post(client, "/mode", {"mode": "auto"}, headers=bearer())
    assert resumed.status_code == 200
    assert resumed.json()["state"] == "safe"
    assert core.submitted == [Pause(), Resume()]


def test_the_alignment_routes_start_and_stop(client: TestClient, core: FakeCoreClient) -> None:
    started = post(client, "/alignment/start", {"exposure_s": 0.25, "gain": 90}, headers=bearer())
    assert started.json()["state"] == "align"
    bare = post(client, "/alignment/start", None, headers=bearer())
    assert bare.status_code == 200
    stopped = post(client, "/alignment/stop", None, headers=bearer())
    assert stopped.json()["state"] == "safe"
    assert core.submitted == [
        StartAlignment(exposure_s=0.25, gain=90),
        StartAlignment(),
        StopAlignment(),
    ]


def test_an_empty_json_object_starts_the_alignment_with_the_defaults(
    client: TestClient, core: FakeCoreClient
) -> None:
    assert post(client, "/alignment/start", {}, headers=bearer()).status_code == 200
    assert core.submitted == [StartAlignment()]


# --- The answers of the scheduler ------------------------------------------------------------


def test_a_rejection_is_a_409_with_the_reason(client: TestClient, core: FakeCoreClient) -> None:
    response = post(client, "/alignment/stop", None, headers=bearer())
    assert response.status_code == 409
    assert response.json() == {
        "accepted": False,
        "message": "no alignment runs",
        "state": "auto",
        "reason": "not_aligning",
        "task_id": None,
    }


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        (RejectReason.PAUSED, 409),
        (RejectReason.ALREADY_PAUSED, 409),
        (RejectReason.NOT_PAUSED, 409),
        (RejectReason.NOT_ALIGNING, 409),
        (RejectReason.DEGRADED, 409),
        (RejectReason.NO_HANDLER, 409),
        (RejectReason.QUEUE_FULL, 409),
        (RejectReason.BUSY, 409),
        (RejectReason.INVALID, 422),
        (RejectReason.CLOSED, 503),
    ],
)
def test_every_rejection_reason_maps_to_a_status(
    client: TestClient, core: FakeCoreClient, reason: RejectReason, status: int
) -> None:
    core.submit = lambda command: CommandResult(  # type: ignore[method-assign]
        accepted=False, message=f"the scheduler said {reason.value}", state="auto", reason=reason
    )
    response = post(client, "/mode", {"mode": "paused"}, headers=bearer())
    assert response.status_code == status
    if status == 409:
        assert response.json()["reason"] == reason.value
    else:
        assert response.json()["error"]["message"] == f"the scheduler said {reason.value}"


def test_the_text_of_a_rejection_loses_its_paths(client: TestClient, core: FakeCoreClient) -> None:
    core.submit = lambda command: CommandResult(  # type: ignore[method-assign]
        accepted=False,
        message="cannot open /home/someone/recordings/a.ser",  # repo-check: allow
        state="auto",
        reason=RejectReason.INVALID,
    )
    response = post(client, "/commands/replay", {"source": "a"}, headers=bearer())
    assert response.status_code == 422
    assert "someone" not in response.text


def test_the_queue_fills_up_and_the_scheduler_says_so(
    client: TestClient, clock: VirtualClock
) -> None:
    results = []
    for _ in range(9):
        results.append(post(client, "/commands/burst", {}, headers=bearer()))
        clock.advance(61)
    assert [r.status_code for r in results] == [200] * 8 + [409]
    assert results[-1].json()["reason"] == "queue_full"


def test_a_core_that_does_not_answer_is_a_503_with_a_retry_time(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.fail_with = CoreUnavailableError("core does not answer")
    response = post(client, "/mode", {"mode": "paused"}, headers=bearer())
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert response.json()["error"]["code"] == "core_unavailable"
    assert "core does not answer" not in response.text  # the message is the server's own


def test_a_core_that_answers_nonsense_is_a_502(client: TestClient, core: FakeCoreClient) -> None:
    core.fail_with = CoreProtocolError("core could not handle submit")
    response = post(client, "/mode", {"mode": "paused"}, headers=bearer())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "core_error"


# --- The validation of the body --------------------------------------------------------------

LONG = "x" * 200


def stream(**fields: Any) -> dict[str, Any]:
    """A burst body with a valid stream, changed by the fields."""
    return {"stream": {"mode": "bin1", "exposure_us": 1, "gain": 0, **fields}}


INVALID: list[tuple[str, Any]] = [
    # the burst
    ("/commands/burst", {"duration_s": 0}),
    ("/commands/burst", {"duration_s": -5}),
    ("/commands/burst", {"duration_s": 601}),
    ("/commands/burst", {"duration_s": "10"}),
    ("/commands/burst", {"duration_s": None}),
    ("/commands/burst", {"duration_s": True}),
    ("/commands/burst", {"label": "a" * 41}),
    ("/commands/burst", {"label": "bad/label"}),
    ("/commands/burst", {"label": "a\nb"}),
    ("/commands/burst", {"label": 5}),
    ("/commands/burst", {"priority": 11}),
    ("/commands/burst", {"priority": -11}),
    ("/commands/burst", {"priority": 1.5}),
    ("/commands/burst", {"unknown": 1}),
    ("/commands/burst", {"stream": {"mode": "bin1"}}),
    ("/commands/burst", {"stream": {"mode": "", "exposure_us": 1, "gain": 0}}),
    ("/commands/burst", {"stream": {"mode": "bin 1", "exposure_us": 1, "gain": 0}}),
    ("/commands/burst", {"stream": {"mode": "a" * 17, "exposure_us": 1, "gain": 0}}),
    ("/commands/burst", {"stream": {"mode": "bin1", "exposure_us": 0, "gain": 0}}),
    ("/commands/burst", {"stream": {"mode": "bin1", "exposure_us": 2_000_000_001, "gain": 0}}),
    ("/commands/burst", {"stream": {"mode": "bin1", "exposure_us": 1, "gain": -1}}),
    ("/commands/burst", {"stream": {"mode": "bin1", "exposure_us": 1, "gain": 1001}}),
    ("/commands/burst", stream(pixel_format="RAW12")),
    ("/commands/burst", stream(roi={"x": 0, "y": 0, "width": 0, "height": 1})),
    ("/commands/burst", stream(roi={"x": -1, "y": 0, "width": 1, "height": 1})),
    ("/commands/burst", stream(extra=1)),
    # the sweep
    ("/commands/sweep", {"exposure_us": list(range(1, 18))}),
    ("/commands/sweep", {"exposure_us": [0]}),
    ("/commands/sweep", {"exposure_us": [1.5]}),
    ("/commands/sweep", {"exposure_us": "500"}),
    ("/commands/sweep", {"gain": [-1]}),
    ("/commands/sweep", {"gain": [1001]}),
    ("/commands/sweep", {"roi_arcmin": [0]}),
    ("/commands/sweep", {"roi_arcmin": [61]}),
    ("/commands/sweep", {"modes": ["bad mode"]}),
    ("/commands/sweep", {"modes": [1]}),
    ("/commands/sweep", {"modes": ["a"] * 17}),
    ("/commands/sweep", {"window_s": 0}),
    ("/commands/sweep", {"window_s": 601}),
    ("/commands/sweep", {"priority": 99}),
    # the replay
    ("/commands/replay", {}),
    ("/commands/replay", {"source": ""}),
    ("/commands/replay", {"source": "../secret"}),
    ("/commands/replay", {"source": "dir/file"}),
    ("/commands/replay", {"source": "dir\\file"}),
    ("/commands/replay", {"source": "C:file"}),
    ("/commands/replay", {"source": "/etc/passwd"}),
    ("/commands/replay", {"source": ".hidden"}),
    ("/commands/replay", {"source": "a" * 129}),
    ("/commands/replay", {"source": "a b"}),
    ("/commands/replay", {"source": "a", "speed": -1}),
    ("/commands/replay", {"source": "a", "speed": 1001}),
    ("/commands/replay", {"source": "a", "speed": "fast"}),
    ("/commands/replay", {"source": "a", "options": {f"k{i}": 1 for i in range(9)}}),
    ("/commands/replay", {"source": "a", "options": {"Bad Key": 1}}),
    ("/commands/replay", {"source": "a", "options": {"k": [1]}}),
    ("/commands/replay", {"source": "a", "options": {"k": {"x": 1}}}),
    ("/commands/replay", {"source": "a", "options": {"k": "v" * 65}}),
    ("/commands/replay", {"source": "a", "options": []}),
    # the dark session
    ("/commands/dark", {"exposure_s": 0}),
    ("/commands/dark", {"exposure_s": -1}),
    ("/commands/dark", {"exposure_s": 121}),
    ("/commands/dark", {"exposure_s": 601}),
    ("/commands/dark", {"exposure_s": "30"}),
    ("/commands/dark", {"exposure_s": True}),
    ("/commands/dark", {"frames": 2}),
    ("/commands/dark", {"frames": 51}),
    ("/commands/dark", {"frames": 3.5}),
    ("/commands/dark", {"frames": "5"}),
    ("/commands/dark", {"frames": True}),
    ("/commands/dark", {"bias_frames": 2}),
    ("/commands/dark", {"bias_frames": 51}),
    ("/commands/dark", {"bias_frames": 3.5}),
    ("/commands/dark", {"wait_for_cover": "yes"}),
    ("/commands/dark", {"wait_for_cover": 1}),
    ("/commands/dark", {"wait_for_cover": None}),
    ("/commands/dark", {"pause_after": "no"}),
    ("/commands/dark", {"pause_after": 0}),
    ("/commands/dark", {"wait_for_cover_timeout_s": 0}),
    ("/commands/dark", {"wait_for_cover_timeout_s": -1}),
    ("/commands/dark", {"wait_for_cover_timeout_s": 7201}),
    ("/commands/dark", {"wait_for_cover_timeout_s": "60"}),
    ("/commands/dark", {"wait_for_cover_timeout_s": True}),
    ("/commands/dark", {"immediate": "yes"}),
    ("/commands/dark", {"immediate": 1}),
    ("/commands/dark", {"immediate": None}),
    ("/commands/dark", {"label": "a" * 41}),
    ("/commands/dark", {"label": "bad/label"}),
    ("/commands/dark", {"label": "a\nb"}),
    ("/commands/dark", {"label": 5}),
    ("/commands/dark", {"priority": 1}),
    ("/commands/dark", {"unknown": 1}),
    # the mode
    ("/mode", {}),
    ("/mode", {"mode": "reboot"}),
    ("/mode", {"mode": "safe"}),
    ("/mode", {"mode": "AUTO"}),
    ("/mode", {"mode": 1}),
    ("/mode", {"mode": None}),
    ("/mode", {"mode": "paused", "extra": 1}),
    # the alignment
    ("/alignment/start", {"exposure_s": 0}),
    ("/alignment/start", {"exposure_s": -1}),
    ("/alignment/start", {"exposure_s": 10.1}),
    ("/alignment/start", {"exposure_s": "0.5"}),
    ("/alignment/start", {"gain": -1}),
    ("/alignment/start", {"gain": 1001}),
    ("/alignment/start", {"gain": 1.5}),
    ("/alignment/start", {"unknown": True}),
]


@pytest.mark.parametrize(
    ("path", "body"), INVALID, ids=[f"{path}-{index}" for index, (path, _) in enumerate(INVALID)]
)
def test_a_body_outside_the_bounds_is_a_422_that_never_reaches_core_and_never_shows_a_trace(
    client: TestClient, core: FakeCoreClient, path: str, body: Any
) -> None:
    response = post(client, path, body, headers=bearer())
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_request"
    assert "Traceback" not in response.text
    assert ".py" not in response.text
    assert core.submitted == []


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"not json",
        b"[1, 2]",
        b'"text"',
        b"null",
        b'{"duration_s": NaN}',
        b'{"duration_s": Infinity}',
        b'{"duration_s": -Infinity}',
        b'{"duration_s": 1e999}',
        b'{"duration_s": 5, "duration_s": 6, "x": }',
        b"\xff\xfe",
    ],
)
def test_a_body_that_is_not_a_json_object_is_a_422(
    client: TestClient, core: FakeCoreClient, raw: bytes
) -> None:
    headers = {**bearer(), "Content-Type": "application/json"}
    response = client.post(f"{API}/commands/burst", content=raw, headers=headers)
    assert response.status_code in {400, 422}
    assert response.json()["error"]["code"] in {"invalid_request", "bad_request"}
    assert "Traceback" not in response.text
    assert core.submitted == []


def test_a_body_without_a_json_content_type_is_a_422(
    client: TestClient, core: FakeCoreClient
) -> None:
    response = client.post(
        f"{API}/commands/burst",
        content=b'{"duration_s": 5}',
        headers={**bearer(), "Content-Type": "text/plain"},
    )
    assert response.status_code == 422
    assert core.submitted == []


def test_a_form_body_is_a_422(client: TestClient, core: FakeCoreClient) -> None:
    response = client.post(f"{API}/commands/burst", data={"duration_s": "5"}, headers=bearer())
    assert response.status_code == 422
    assert core.submitted == []


def test_an_invalid_body_with_a_valid_token_counts_against_the_command_limit_only(
    client: TestClient, core: FakeCoreClient
) -> None:
    for _ in range(5):
        assert post(client, "/mode", {"mode": "x"}, headers=bearer()).status_code == 422
    assert post(client, "/mode", {"mode": "paused"}, headers=bearer()).status_code == 429


def test_a_get_on_a_command_route_is_a_405_with_the_error_shape(client: TestClient) -> None:
    response = client.get(f"{API}/commands/burst")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


# --- The size of the body --------------------------------------------------------------------


def test_a_body_above_the_limit_is_a_413(client: TestClient, core: FakeCoreClient) -> None:
    body = {"source": "a", "options": {"k": "v" * 60}, "padding": "x" * 20_000}
    response = post(client, "/commands/replay", body, headers=bearer())
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert core.submitted == []


def test_a_streamed_body_above_the_limit_is_a_413_too(
    client: TestClient, core: FakeCoreClient
) -> None:
    def chunks() -> Any:
        for _ in range(40):
            yield b" " * 1000

    response = client.post(
        f"{API}/commands/burst",
        content=chunks(),
        headers={**bearer(), "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert core.submitted == []


def test_a_body_exactly_at_the_limit_is_read(client: TestClient, core: FakeCoreClient) -> None:
    padding = b" " * (16 * 1024 - len(b"{}"))
    response = client.post(
        f"{API}/commands/burst",
        content=b"{}" + padding,
        headers={**bearer(), "Content-Type": "application/json"},
    )
    assert response.status_code == 200
    assert core.submitted == [QueueBurst()]
