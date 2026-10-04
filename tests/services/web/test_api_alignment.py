"""The alignment routes: the state, the polled frame, and the WebSocket live view."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from seeingmon.clock import VirtualClock
from seeingmon.scheduler.commands import StartAlignment
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import AlignmentFrame
from seeingmon.services.web.core_client import CoreUnavailableError, FakeCoreClient
from seeingmon.store.db import Store
from tests.services.conftest import wait_until
from tests.services.web.client import TestClient
from tests.services.web.helpers import TOKEN, alignment_state, bearer, tiny_jpeg

API = "/api/v1"
STREAM = f"{API}/alignment/stream"


def frame(seq: int) -> AlignmentFrame:
    return AlignmentFrame(alignment_state(seq), tiny_jpeg(seq * 10))


def frames(count: int) -> Callable[[], AsyncIterator[AlignmentFrame]]:
    """A source of `count` frames. After the last one, the stream stays open and silent."""

    async def source() -> AsyncIterator[AlignmentFrame]:
        for seq in range(1, count + 1):
            yield frame(seq)
        await asyncio.Event().wait()

    return source


async def short_sleep(seconds: float) -> None:
    """Stands in for `asyncio.sleep` in the hub: the waits of a test stay short."""
    await asyncio.sleep(min(seconds, 0.01))


@pytest.fixture
def streaming_core(clock: VirtualClock) -> FakeCoreClient:
    return FakeCoreClient(clock=clock, frames=frames(3))


@pytest.fixture
def stream_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    streaming_core: FakeCoreClient,
) -> TestClient:
    return open_client(make_app(core=streaming_core, sleep=short_sleep))


def read_frame(session: WebSocketTestSession) -> tuple[dict[str, Any], bytes]:
    """Read the text message with the state, and the binary message with the JPEG."""
    message = session.receive_json()
    assert message["type"] == "state"
    return message["state"], session.receive_bytes()


def read_newest(session: WebSocketTestSession, seq: int) -> tuple[dict[str, Any], bytes]:
    """Read frames until the one with this sequence number. A viewer may skip the frames before."""
    for _ in range(10):
        state, jpeg = read_frame(session)
        if state["frame"]["seq"] == seq:
            return state, jpeg
    raise AssertionError(f"frame {seq} never came")


# --- The state -------------------------------------------------------------------------------


def test_the_state_is_inactive_outside_alignment(client: TestClient) -> None:
    response = client.get(f"{API}/alignment/state")
    assert response.status_code == 200
    assert response.json() == {
        "active": False,
        "t_utc": None,
        "frame": None,
        "target": None,
        "solved": None,
        "offset": None,
        "focus": None,
        "histogram": None,
        "saturation": None,
        "reticle": None,
        "sky": None,
        "timing": None,
        "quality": {},
    }


def test_the_state_shows_the_offset_the_rotation_the_focus_and_the_histogram(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.submit(StartAlignment())
    core.set_alignment_state(alignment_state(5))
    body = client.get(f"{API}/alignment/state").json()
    assert body["active"] is True
    assert body["frame"]["seq"] == 5
    assert body["target"] == {"x_px": 2072.0, "y_px": 1411.0, "roll_deg": 10.0}
    assert body["offset"]["distance_arcsec"] == 38.2
    assert body["offset"]["roll_deg"] == 0.5
    assert body["focus"]["best_fwhm_px"] == 2.1
    assert body["histogram"]["counts"] == [900, 300, 90, 20, 5, 1, 0, 2]
    assert body["saturation"] == {"fraction": 0.0004, "warning": False}


def test_the_state_is_a_503_when_core_does_not_answer(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.fail_with = CoreUnavailableError("gone")
    response = client.get(f"{API}/alignment/state")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "core_unavailable"


# --- The polled frame ------------------------------------------------------------------------


def test_the_poll_starts_the_stream_and_then_returns_the_newest_frame(
    stream_client: TestClient, streaming_core: FakeCoreClient
) -> None:
    first = stream_client.get(f"{API}/alignment/frame")
    assert first.status_code in {200, 204}  # the stream starts with this request

    def newest() -> bool:
        response = stream_client.get(f"{API}/alignment/frame")
        return bool(response.status_code == 200 and response.headers["x-frame-seq"] == "3")

    assert wait_until(newest)
    response = stream_client.get(f"{API}/alignment/frame")
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == tiny_jpeg(30)
    assert streaming_core.streams_opened == 1


def test_the_poll_returns_204_when_nothing_is_newer(
    stream_client: TestClient,
) -> None:
    def has_frames() -> bool:
        return bool(stream_client.get(f"{API}/alignment/frame").status_code == 200)

    assert wait_until(has_frames)
    sequence = int(stream_client.get(f"{API}/alignment/frame").headers["x-frame-seq"])
    response = stream_client.get(f"{API}/alignment/frame", params={"after": sequence})
    assert response.status_code == 204
    assert response.content == b""


def test_the_poll_without_a_stream_is_a_204(client: TestClient) -> None:
    assert client.get(f"{API}/alignment/frame").status_code == 204


@pytest.mark.parametrize("after", ["-1", "x", "1.5"])
def test_a_bad_after_is_a_422(client: TestClient, after: str) -> None:
    assert client.get(f"{API}/alignment/frame", params={"after": after}).status_code == 422


# --- The WebSocket ---------------------------------------------------------------------------


def test_the_live_view_sends_the_state_and_then_the_jpeg_of_each_frame(
    stream_client: TestClient,
) -> None:
    with stream_client.websocket_connect(STREAM) as session:
        state, jpeg = read_newest(session, 3)
    assert state == alignment_state(3).model_dump(mode="json")
    assert jpeg == tiny_jpeg(30)


def test_the_live_view_sends_the_newest_frame_and_may_skip_the_older_ones(
    stream_client: TestClient,
) -> None:
    with stream_client.websocket_connect(STREAM) as session:
        seen = []
        for _ in range(3):
            state, _jpeg = read_frame(session)
            seen.append(state["frame"]["seq"])
            if seen[-1] == 3:
                break
    assert seen == sorted(seen)
    assert seen[-1] == 3


def test_a_viewer_that_leaves_does_not_stop_the_next_one(
    stream_client: TestClient, streaming_core: FakeCoreClient
) -> None:
    for _ in range(3):
        with stream_client.websocket_connect(STREAM) as session:
            state, jpeg = read_newest(session, 3)
            assert jpeg == tiny_jpeg(30)
            assert state["active"] is True
    assert streaming_core.streams_opened == 1  # one stream to core served all three viewers


def test_a_viewer_that_vanishes_without_closing_is_dropped(
    stream_client: TestClient, streaming_core: FakeCoreClient
) -> None:
    ctx = stream_client.app.state.ctx  # type: ignore[attr-defined]
    session = stream_client.websocket_connect(STREAM)
    session.__enter__()
    read_newest(session, 3)
    assert ctx.hub.viewers == 1
    session.exit_stack.close()  # the client goes away
    assert wait_until(lambda: ctx.hub.viewers == 0)


def test_the_stream_to_core_closes_after_the_last_viewer_has_gone_for_the_idle_time(
    stream_client: TestClient, streaming_core: FakeCoreClient, clock: VirtualClock
) -> None:
    ctx = stream_client.app.state.ctx  # type: ignore[attr-defined]
    with stream_client.websocket_connect(STREAM) as session:
        read_newest(session, 3)
    assert wait_until(lambda: ctx.hub.viewers == 0)
    clock.advance(6)  # the idle time of the test settings is 5 s
    assert stream_client.portal is not None
    assert stream_client.portal.call(ctx.hub.check_idle) is True
    assert wait_until(lambda: streaming_core.streams_closed == 1)


def test_a_new_viewer_after_that_opens_the_stream_again(
    stream_client: TestClient, streaming_core: FakeCoreClient, clock: VirtualClock
) -> None:
    ctx = stream_client.app.state.ctx  # type: ignore[attr-defined]
    with stream_client.websocket_connect(STREAM) as session:
        read_newest(session, 3)
    assert wait_until(lambda: ctx.hub.viewers == 0)
    clock.advance(6)
    assert stream_client.portal is not None
    stream_client.portal.call(ctx.hub.check_idle)
    with stream_client.websocket_connect(STREAM) as session:
        read_newest(session, 3)
    assert streaming_core.streams_opened == 2


def test_the_live_view_says_idle_while_no_frame_arrives(
    make_app: Callable[..., FastAPI], open_client: Callable[..., TestClient], seeded: Store
) -> None:
    quiet = FakeCoreClient()  # a stream that sends nothing
    client = open_client(make_app(core=quiet, sleep=short_sleep))
    with client.websocket_connect(STREAM) as session:
        assert session.receive_json() == {"type": "idle"}
        assert session.receive_json() == {"type": "idle"}


def test_the_live_view_reports_a_broken_stream_and_goes_on_when_it_recovers(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    core = FakeCoreClient(clock=clock, frames=frames(1))
    core.fail_with = CoreUnavailableError("gone")
    client = open_client(make_app(core=core, sleep=short_sleep))
    with client.websocket_connect(STREAM) as session:
        assert session.receive_json() == {"type": "error", "code": "core_unavailable"}
        core.fail_with = None  # core is back, and the hub reconnects by itself
        state, jpeg = read_newest(session, 1)
        assert jpeg == tiny_jpeg(10)
        assert state["frame"]["seq"] == 1


def test_a_viewer_over_the_limit_is_accepted_and_then_told_to_try_later(
    stream_client: TestClient,
) -> None:
    """The handshake completes first. A close before it becomes an HTTP 403, and a browser reports
    that as code 1006, so the page could not tell a full server from a broken one."""
    with stream_client.websocket_connect(STREAM) as first, stream_client.websocket_connect(STREAM):
        read_newest(first, 3)
        with stream_client.websocket_connect(STREAM) as turned_away:
            with pytest.raises(WebSocketDisconnect) as raised:
                turned_away.receive_json()
            assert raised.value.code == 1013
    with stream_client.websocket_connect(STREAM) as again:  # a place is free again
        read_newest(again, 3)


def test_a_viewer_that_was_turned_away_does_not_count_as_a_viewer(
    stream_client: TestClient,
) -> None:
    ctx = stream_client.app.state.ctx  # type: ignore[attr-defined]
    with stream_client.websocket_connect(STREAM) as first, stream_client.websocket_connect(STREAM):
        read_newest(first, 3)
        for _ in range(3):
            with (
                stream_client.websocket_connect(STREAM) as turned_away,
                pytest.raises(WebSocketDisconnect),
            ):
                turned_away.receive_json()
        assert ctx.hub.viewers == 2


def test_a_websocket_to_an_unknown_path_is_refused(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"{API}/alignment/nothing"),
    ):
        pass


# --- The token for reads ---------------------------------------------------------------------


@pytest.fixture
def locked_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    streaming_core: FakeCoreClient,
) -> TestClient:
    settings = WebSettings.model_validate(
        {
            "require_token_for_reads": True,
            "live": {"max_fps": 30.0, "stall_s": 0.3, "idle_s": 5.0, "max_clients": 2},
        }
    )
    return open_client(make_app(settings=settings, core=streaming_core, sleep=short_sleep))


def test_a_client_that_needs_a_token_sends_it_in_its_first_message(
    locked_client: TestClient,
) -> None:
    with locked_client.websocket_connect(STREAM) as session:
        session.send_json({"type": "auth", "token": TOKEN})
        state, jpeg = read_newest(session, 3)
        assert jpeg == tiny_jpeg(30)
        assert state["active"] is True


def test_a_client_that_needs_a_token_may_send_it_as_a_header(locked_client: TestClient) -> None:
    with locked_client.websocket_connect(STREAM, headers=bearer()) as session:
        read_newest(session, 3)


@pytest.mark.parametrize(
    "message",
    [
        {"type": "auth", "token": "wrong"},
        {"type": "auth"},
        {"type": "auth", "token": 5},
        {"type": "hello", "token": TOKEN},
        {"token": TOKEN},
        ["auth", TOKEN],
    ],
)
def test_a_wrong_first_message_closes_the_stream_with_the_policy_code(
    locked_client: TestClient, message: Any
) -> None:
    with locked_client.websocket_connect(STREAM) as session:
        session.send_json(message)
        with pytest.raises(WebSocketDisconnect) as raised:
            session.receive_json()
        assert raised.value.code == 1008


def test_a_full_server_asks_for_the_token_before_it_says_try_later(
    locked_client: TestClient,
) -> None:
    """A viewer without the token learns nothing about the load of the server."""
    with (
        locked_client.websocket_connect(STREAM) as one,
        locked_client.websocket_connect(STREAM) as two,
    ):
        for session in (one, two):
            session.send_json({"type": "auth", "token": TOKEN})
            read_newest(session, 3)
        with locked_client.websocket_connect(STREAM) as with_token:
            with_token.send_json({"type": "auth", "token": TOKEN})
            with pytest.raises(WebSocketDisconnect) as busy:
                with_token.receive_json()
            assert busy.value.code == 1013
        with locked_client.websocket_connect(STREAM) as without_token:
            without_token.send_json({"type": "auth", "token": "wrong"})
            with pytest.raises(WebSocketDisconnect) as denied:
                without_token.receive_json()
            assert denied.value.code == 1008


def test_a_wrong_header_is_refused_before_the_handshake(locked_client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        locked_client.websocket_connect(STREAM, headers=bearer("wrong")),
    ):
        pass


def test_the_state_and_the_poll_need_the_token_too(locked_client: TestClient) -> None:
    assert locked_client.get(f"{API}/alignment/state").status_code == 401
    assert locked_client.get(f"{API}/alignment/frame").status_code == 401
    assert locked_client.get(f"{API}/alignment/state", headers=bearer()).status_code == 200
    assert locked_client.get(f"{API}/alignment/frame", headers=bearer()).status_code in {200, 204}
