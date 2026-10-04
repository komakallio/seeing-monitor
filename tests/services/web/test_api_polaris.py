"""The routes of the live video of Polaris and of the rolling seeing value.

The WebSocket and the polled frame share the code of the alignment live view, and
`test_api_alignment.py` proves that part. These tests prove what the video adds: its own hub, its
own cap of frames per second, the sequence number that the hub stamps, the PNG, and the state in
the header of the polled frame.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from seeingmon.clock import VirtualClock, utc_ns_to_iso
from seeingmon.services.web.config import WebSettings
from seeingmon.services.web.contract import AlignmentFrame, PolarisFrame
from seeingmon.services.web.core_client import CoreUnavailableError, FakeCoreClient
from seeingmon.store.db import Store
from tests.services.conftest import wait_until
from tests.services.web.client import TestClient
from tests.services.web.helpers import (
    TOKEN,
    alignment_state,
    bearer,
    live_seeing_view,
    polaris_state,
    tiny_jpeg,
    tiny_png,
)

API = "/api/v1"
STREAM = f"{API}/polaris/stream"
FRAME = f"{API}/polaris/frame"
LIVE = f"{API}/seeing/live"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def frame(index: int) -> PolarisFrame:
    """A frame as `core` sends it, with `seq` 0. The hub stamps the sequence number."""
    return PolarisFrame(polaris_state(seq=0), tiny_png(index * 10))


def frames(count: int) -> Callable[[], AsyncIterator[PolarisFrame]]:
    """A source of `count` frames. After the last one, the stream stays open and silent."""

    async def source() -> AsyncIterator[PolarisFrame]:
        for index in range(1, count + 1):
            yield frame(index)
        await asyncio.Event().wait()

    return source


async def short_sleep(seconds: float) -> None:
    """Stands in for `asyncio.sleep` in the hub: the waits of a test stay short."""
    await asyncio.sleep(min(seconds, 0.01))


@pytest.fixture
def streaming_core(clock: VirtualClock) -> FakeCoreClient:
    return FakeCoreClient(clock=clock, polaris=frames(3))


@pytest.fixture
def stream_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    streaming_core: FakeCoreClient,
) -> TestClient:
    return open_client(make_app(core=streaming_core, sleep=short_sleep))


def read_frame(session: WebSocketTestSession) -> tuple[dict[str, Any], bytes]:
    """Read the text message with the state, and the binary message with the image."""
    message = session.receive_json()
    assert message["type"] == "state"
    return message["state"], session.receive_bytes()


def read_newest(session: WebSocketTestSession, seq: int) -> tuple[dict[str, Any], bytes]:
    """Read frames until the one with this sequence number. A viewer may skip the frames before."""
    for _ in range(10):
        state, image = read_frame(session)
        if state["seq"] == seq:
            return state, image
    raise AssertionError(f"frame {seq} never came")


# --- The WebSocket ---------------------------------------------------------------------------


def test_the_video_sends_the_state_and_then_the_png_of_each_frame(
    stream_client: TestClient,
) -> None:
    with stream_client.websocket_connect(STREAM) as session:
        state, image = read_newest(session, 3)
    assert state == polaris_state(seq=3).model_dump(mode="json")
    assert image == tiny_png(30)
    assert image.startswith(PNG_MAGIC)
    assert state["image_type"] == "image/png"


def test_the_sequence_number_of_the_state_is_the_one_of_the_hub(stream_client: TestClient) -> None:
    with stream_client.websocket_connect(STREAM) as session:
        seen = []
        for _ in range(3):
            state, _image = read_frame(session)
            seen.append(state["seq"])
            if seen[-1] == 3:
                break
    assert seen == sorted(seen)
    assert seen[-1] == 3
    assert set(seen) <= {1, 2, 3}


def test_a_viewer_that_leaves_does_not_stop_the_next_one(
    stream_client: TestClient, streaming_core: FakeCoreClient
) -> None:
    for _ in range(3):
        with stream_client.websocket_connect(STREAM) as session:
            _state, image = read_newest(session, 3)
            assert image == tiny_png(30)
    assert streaming_core.polaris_streams_opened == 1  # one stream to core served all three


def test_the_video_does_not_open_the_alignment_stream_of_core(
    stream_client: TestClient, streaming_core: FakeCoreClient
) -> None:
    with stream_client.websocket_connect(STREAM) as session:
        read_newest(session, 3)
    assert streaming_core.streams_opened == 0


def test_the_video_has_its_own_viewers_and_the_alignment_view_does_not_count_against_it(
    stream_client: TestClient,
) -> None:
    ctx = stream_client.app.state.ctx  # type: ignore[attr-defined]
    with (
        stream_client.websocket_connect(f"{API}/alignment/stream"),
        stream_client.websocket_connect(f"{API}/alignment/stream"),
    ):
        assert wait_until(lambda: ctx.hub.viewers == 2)
        assert ctx.polaris_hub.viewers == 0
        with stream_client.websocket_connect(STREAM) as session:  # the alignment view is full
            read_newest(session, 3)
        assert ctx.hub.viewers == 2


def test_a_viewer_over_the_limit_is_accepted_and_then_told_to_try_later(
    stream_client: TestClient,
) -> None:
    with stream_client.websocket_connect(STREAM) as first, stream_client.websocket_connect(STREAM):
        read_newest(first, 3)
        with stream_client.websocket_connect(STREAM) as turned_away:
            with pytest.raises(WebSocketDisconnect) as raised:
                turned_away.receive_json()
            assert raised.value.code == 1013
    with stream_client.websocket_connect(STREAM) as again:  # a place is free again
        read_newest(again, 3)


def test_a_viewer_that_vanishes_without_closing_is_dropped(stream_client: TestClient) -> None:
    ctx = stream_client.app.state.ctx  # type: ignore[attr-defined]
    session = stream_client.websocket_connect(STREAM)
    session.__enter__()
    read_newest(session, 3)
    assert ctx.polaris_hub.viewers == 1
    session.exit_stack.close()  # the client goes away
    assert wait_until(lambda: ctx.polaris_hub.viewers == 0)


def test_the_stream_to_core_closes_after_the_last_viewer_has_gone_for_the_idle_time(
    stream_client: TestClient, streaming_core: FakeCoreClient, clock: VirtualClock
) -> None:
    ctx = stream_client.app.state.ctx  # type: ignore[attr-defined]
    with stream_client.websocket_connect(STREAM) as session:
        read_newest(session, 3)
    assert wait_until(lambda: ctx.polaris_hub.viewers == 0)
    clock.advance(6)  # the idle time of the test settings is 5 s
    assert stream_client.portal is not None
    assert stream_client.portal.call(ctx.polaris_hub.check_idle) is True
    assert wait_until(lambda: streaming_core.polaris_streams_closed == 1)


def test_the_video_says_idle_while_no_frame_arrives(
    make_app: Callable[..., FastAPI], open_client: Callable[..., TestClient], seeded: Store
) -> None:
    client = open_client(make_app(core=FakeCoreClient(), sleep=short_sleep))  # a silent stream
    with client.websocket_connect(STREAM) as session:
        assert session.receive_json() == {"type": "idle"}
        assert session.receive_json() == {"type": "idle"}


def test_the_video_reports_a_broken_stream_and_goes_on_when_it_recovers(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    core = FakeCoreClient(clock=clock, polaris=frames(1))
    core.fail_with = CoreUnavailableError("gone")
    client = open_client(make_app(core=core, sleep=short_sleep))
    with client.websocket_connect(STREAM) as session:
        assert session.receive_json() == {"type": "error", "code": "core_unavailable"}
        core.fail_with = None  # core is back, and the hub reconnects by itself
        state, image = read_newest(session, 1)
        assert image == tiny_png(10)
        assert state["seq"] == 1


def spaced(make: Callable[[int], Any], count: int) -> Callable[[], AsyncIterator[Any]]:
    """A source that yields `count` frames 20 ms apart, so that a viewer sees them one by one."""

    async def source() -> AsyncIterator[Any]:
        for index in range(1, count + 1):
            await asyncio.sleep(0.02)
            yield make(index)
        await asyncio.Event().wait()

    return source


def test_each_view_keeps_its_own_cap_of_frames_per_second(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
) -> None:
    """The clock of a test stands still, so after its first frame a viewer waits one interval."""
    waits: list[float] = []

    async def recording_sleep(seconds: float) -> None:
        waits.append(seconds)
        await asyncio.sleep(0.001)

    core = FakeCoreClient(
        clock=clock,
        frames=spaced(lambda index: AlignmentFrame(alignment_state(index), tiny_jpeg(index)), 8),
        polaris=spaced(frame, 8),
    )
    settings = WebSettings.model_validate(
        {"live": {"max_fps": 4.0, "polaris_max_fps": 25.0, "stall_s": 5.0, "max_clients": 2}}
    )
    client = open_client(make_app(settings=settings, core=core, sleep=recording_sleep))

    def viewer_waits() -> list[float]:
        return [wait for wait in waits if wait < 1.0]  # the idle ticker sleeps for a second

    with client.websocket_connect(STREAM) as session:
        for _ in range(3):
            read_frame(session)
    assert viewer_waits()
    assert viewer_waits() == [pytest.approx(1 / 25.0)] * len(viewer_waits())
    waits.clear()
    with client.websocket_connect(f"{API}/alignment/stream") as session:
        for _ in range(3):
            message = session.receive_json()
            assert message["type"] == "state"
            session.receive_bytes()
    assert viewer_waits()
    assert viewer_waits() == [pytest.approx(1 / 4.0)] * len(viewer_waits())


# --- The polled frame ------------------------------------------------------------------------


def test_the_poll_starts_the_stream_and_then_returns_the_newest_frame(
    stream_client: TestClient, streaming_core: FakeCoreClient
) -> None:
    first = stream_client.get(FRAME)
    assert first.status_code in {200, 204}  # the stream starts with this request

    def newest() -> bool:
        response = stream_client.get(FRAME)
        return bool(response.status_code == 200 and response.headers["x-frame-seq"] == "3")

    assert wait_until(newest)
    response = stream_client.get(FRAME)
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == tiny_png(30)
    assert streaming_core.polaris_streams_opened == 1
    assert streaming_core.streams_opened == 0


def test_the_header_of_the_polled_frame_holds_its_state(stream_client: TestClient) -> None:
    def has_frames() -> bool:
        return bool(stream_client.get(FRAME).status_code == 200)

    assert wait_until(has_frames)
    assert wait_until(lambda: stream_client.get(FRAME).headers.get("x-frame-seq") == "3")
    response = stream_client.get(FRAME)
    header = response.headers["x-frame-state"]
    assert "\n" not in header
    state = json.loads(header)
    assert state == polaris_state(seq=3).model_dump(mode="json")
    assert state["seq"] == int(response.headers["x-frame-seq"])
    assert state["image_type"] == "image/png"
    assert response.headers["content-type"] == state["image_type"]
    assert (state["image_width"], state["image_height"]) == (128, 128)


def test_the_poll_returns_204_when_nothing_is_newer(stream_client: TestClient) -> None:
    def has_frames() -> bool:
        return bool(stream_client.get(FRAME).status_code == 200)

    assert wait_until(has_frames)
    sequence = int(stream_client.get(FRAME).headers["x-frame-seq"])
    response = stream_client.get(FRAME, params={"after": sequence})
    assert response.status_code == 204
    assert response.content == b""


def test_a_poll_that_is_ahead_of_the_server_gets_the_newest_frame(
    stream_client: TestClient,
) -> None:
    """After a restart of the web process the numbers start again at 1, and a page that polls
    keeps its old number. The server answers it with the newest frame, so the page never stalls."""

    def has_frames() -> bool:
        return bool(stream_client.get(FRAME).status_code == 200)

    assert wait_until(has_frames)
    assert wait_until(lambda: stream_client.get(FRAME).headers.get("x-frame-seq") == "3")
    response = stream_client.get(FRAME, params={"after": 40_000})
    assert response.status_code == 200
    assert response.headers["x-frame-seq"] == "3"


def test_the_poll_without_a_stream_is_a_204(client: TestClient) -> None:
    assert client.get(FRAME).status_code == 204


@pytest.mark.parametrize("after", ["-1", "x", "1.5"])
def test_a_bad_after_is_a_422(client: TestClient, after: str) -> None:
    assert client.get(FRAME, params={"after": after}).status_code == 422


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
    core = streaming_core
    core.live = live_seeing_view()
    return open_client(make_app(settings=settings, core=core, sleep=short_sleep))


def test_a_client_that_needs_a_token_sends_it_in_its_first_message(
    locked_client: TestClient,
) -> None:
    with locked_client.websocket_connect(STREAM) as session:
        session.send_json({"type": "auth", "token": TOKEN})
        _state, image = read_newest(session, 3)
        assert image == tiny_png(30)


def test_a_client_that_needs_a_token_may_send_it_as_a_header(locked_client: TestClient) -> None:
    with locked_client.websocket_connect(STREAM, headers=bearer()) as session:
        read_newest(session, 3)


@pytest.mark.parametrize(
    "message",
    [
        {"type": "auth", "token": "wrong"},
        {"type": "auth"},
        {"type": "hello", "token": TOKEN},
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


def test_the_poll_and_the_live_value_need_the_token_too(locked_client: TestClient) -> None:
    assert locked_client.get(FRAME).status_code == 401
    assert locked_client.get(LIVE).status_code == 401
    assert locked_client.get(FRAME, headers=bearer()).status_code in {200, 204}
    assert locked_client.get(LIVE, headers=bearer()).status_code == 200


# --- The rolling seeing value ----------------------------------------------------------------


def test_the_rolling_seeing_value_carries_its_time_and_its_age(
    client: TestClient, core: FakeCoreClient, clock: VirtualClock
) -> None:
    now_ns = clock.utc_ns()
    core.live = live_seeing_view(t_utc_ns=now_ns - 2_500_000_000)
    response = client.get(LIVE)
    assert response.status_code == 200
    body = response.json()
    assert body["age_s"] == 2.5
    assert body["t_utc"] == utc_ns_to_iso(now_ns - 2_500_000_000)
    assert body["seeing_fwhm_arcsec"] == 1.62
    assert body["r0_cm"] == 6.1
    assert body["n_frames"] == 820
    assert body["flags"] == []
    assert body["quality"] == {}
    clock.advance(10)  # no new frames: the value ages
    assert client.get(LIVE).json()["age_s"] == 12.5


def test_the_age_never_goes_below_zero(
    client: TestClient, core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.live = live_seeing_view(t_utc_ns=clock.utc_ns() + 5_000_000_000)
    assert client.get(LIVE).json()["age_s"] == 0.0


def test_a_value_with_null_fields_says_why(
    client: TestClient, core: FakeCoreClient, clock: VirtualClock
) -> None:
    core.live = live_seeing_view(
        t_utc_ns=clock.utc_ns(),
        seeing_fwhm_arcsec=None,
        r0_cm=None,
        flags=["cloud", "degraded"],
        quality={"seeing_fwhm_arcsec": "too few usable frames", "r0_cm": "too few usable frames"},
    )
    body = client.get(LIVE).json()
    assert body["seeing_fwhm_arcsec"] is None
    assert body["r0_cm"] is None
    assert body["flags"] == ["cloud", "degraded"]
    assert body["quality"]["r0_cm"] == "too few usable frames"


def test_without_a_value_the_answer_is_404_no_data(client: TestClient) -> None:
    response = client.get(LIVE)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_data"


def test_the_rolling_seeing_value_is_a_503_when_core_does_not_answer(
    client: TestClient, core: FakeCoreClient
) -> None:
    core.fail_with = CoreUnavailableError("gone")
    response = client.get(LIVE)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "core_unavailable"
