"""The focus history on the live view, and the reset of the best focus value."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.testclient import WebSocketTestSession

from seeingmon.clock import VirtualClock
from seeingmon.services.web.contract import (
    AlignmentFrame,
    AlignmentState,
    FocusHistoryView,
    FocusView,
)
from seeingmon.services.web.core_client import CoreUnavailableError, FakeCoreClient
from seeingmon.services.web.live import HISTORY_LISTS, HistoryCursor
from seeingmon.store.db import Store
from tests.services.web.client import TestClient
from tests.services.web.helpers import TOKEN, bearer, tiny_jpeg

API = "/api/v1"
STREAM = f"{API}/alignment/stream"


def history(indexes: range, session: int = 1) -> FocusHistoryView:
    """The history of the points `indexes`, one point for each frame, 0.5 s apart."""
    points = list(indexes)
    return FocusHistoryView(
        session=session,
        index=points,
        seq=[100 + n for n in points],
        t_utc_ms=[1_800_000_000_000 + 500 * n for n in points],
        fwhm_px=[2.0 + 0.01 * n for n in points],
        fwhm_arcsec=[round((2.0 + 0.01 * n) * 3.82, 4) for n in points],
        n_stars=[30] * len(points),
        spike=[n % 5 == 0 for n in points],
    )


def state_with(indexes: range, session: int = 1) -> dict[str, Any]:
    """The JSON of a state whose focus holds the history of `indexes`."""
    focus = FocusView(fwhm_px=2.5, best_fwhm_px=2.0, history=history(indexes, session))
    return AlignmentState(active=True, focus=focus).model_dump(mode="json")


class TestTheCursorOfAViewer:
    def test_the_first_message_holds_the_whole_history_and_says_reset(self) -> None:
        state = state_with(range(1, 6))
        HistoryCursor().delta(state)
        assert state["focus"]["history"]["reset"] is True
        assert state["focus"]["history"]["index"] == [1, 2, 3, 4, 5]

    def test_the_next_message_holds_only_the_new_points(self) -> None:
        cursor = HistoryCursor()
        cursor.delta(state_with(range(1, 6)))
        state = state_with(range(2, 8))  # the history is a window: the oldest point left
        cursor.delta(state)
        history = state["focus"]["history"]
        assert history["reset"] is False
        assert history["index"] == [6, 7]
        for name in HISTORY_LISTS:
            assert len(history[name]) == 2
        assert history["seq"] == [106, 107]
        assert history["spike"] == [False, False]

    def test_a_message_without_new_points_holds_empty_lists(self) -> None:
        cursor = HistoryCursor()
        cursor.delta(state_with(range(1, 6)))
        state = state_with(range(1, 6))
        cursor.delta(state)
        history = state["focus"]["history"]
        assert history["reset"] is False
        assert all(history[name] == [] for name in HISTORY_LISTS)

    def test_a_viewer_that_missed_states_gets_every_point_that_it_lacks(self) -> None:
        cursor = HistoryCursor()
        cursor.delta(state_with(range(1, 4)))
        state = state_with(range(1, 12))  # the viewer skipped several frames
        cursor.delta(state)
        assert state["focus"]["history"]["index"] == list(range(4, 12))

    def test_a_new_session_resets_the_viewer(self) -> None:
        cursor = HistoryCursor()
        cursor.delta(state_with(range(1, 30), session=1))
        state = state_with(range(1, 4), session=2)
        cursor.delta(state)
        assert state["focus"]["history"]["reset"] is True
        assert state["focus"]["history"]["index"] == [1, 2, 3]

    def test_a_history_that_runs_backward_resets_the_viewer(self) -> None:
        cursor = HistoryCursor()
        cursor.delta(state_with(range(1, 30), session=0))
        state = state_with(range(1, 4), session=0)  # core restarted: the same session number
        cursor.delta(state)
        assert state["focus"]["history"]["reset"] is True
        assert state["focus"]["history"]["index"] == [1, 2, 3]

    def test_a_state_without_a_history_stays_as_it_is(self) -> None:
        state = AlignmentState(active=True, focus=FocusView(fwhm_px=2.5)).model_dump(mode="json")
        before = json.dumps(state)
        HistoryCursor().delta(state)
        assert json.dumps(state) == before
        inactive = AlignmentState(active=False).model_dump(mode="json")
        HistoryCursor().delta(inactive)
        assert inactive["focus"] is None

    def test_two_viewers_keep_their_own_cursors(self) -> None:
        one, two = HistoryCursor(), HistoryCursor()
        one.delta(state_with(range(1, 6)))
        later = state_with(range(1, 9))
        again = state_with(range(1, 9))
        one.delta(later)
        two.delta(again)
        assert later["focus"]["history"]["index"] == [6, 7, 8]
        assert again["focus"]["history"]["index"] == list(range(1, 9))
        assert again["focus"]["history"]["reset"] is True


class Gate:
    """Lets a test hold a source back until the viewer has read what came before."""

    def __init__(self) -> None:
        self.open = threading.Event()


def gated_frames(gate: Gate) -> Callable[[], AsyncIterator[AlignmentFrame]]:
    """Frames with the points 1 to 3, then (after the gate opens) the points 1 to 5."""

    def frame(last: int) -> AlignmentFrame:
        focus = FocusView(fwhm_px=2.5, history=history(range(1, last + 1)))
        return AlignmentFrame(AlignmentState(active=True, focus=focus), tiny_jpeg(last * 10))

    async def source() -> AsyncIterator[AlignmentFrame]:
        yield frame(3)
        while not gate.open.is_set():
            await asyncio.sleep(0.005)
        yield frame(5)
        await asyncio.Event().wait()

    return source


async def short_sleep(seconds: float) -> None:
    await asyncio.sleep(min(seconds, 0.01))


@pytest.fixture
def gate() -> Gate:
    return Gate()


@pytest.fixture
def stream_client(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    clock: VirtualClock,
    gate: Gate,
) -> TestClient:
    core = FakeCoreClient(clock=clock, frames=gated_frames(gate))
    return open_client(make_app(core=core, sleep=short_sleep))


def read_history(session: WebSocketTestSession) -> dict[str, Any]:
    message = session.receive_json()
    assert message["type"] == "state"
    session.receive_bytes()
    return message["state"]["focus"]["history"]  # type: ignore[no-any-return]


def test_a_viewer_gets_the_whole_history_first_and_then_only_the_new_points(
    stream_client: TestClient, gate: Gate
) -> None:
    with stream_client.websocket_connect(STREAM) as session:
        first = read_history(session)
        assert first["reset"] is True
        assert first["index"] == [1, 2, 3]
        assert first["fwhm_arcsec"] == [round((2.0 + 0.01 * n) * 3.82, 4) for n in (1, 2, 3)]
        gate.open.set()
        later = read_history(session)
        while later["index"] == []:  # a message without a new point may come first
            later = read_history(session)
        assert later["reset"] is False
        assert later["index"] == [4, 5]
        assert later["seq"] == [104, 105]
        assert later["spike"] == [False, True]  # the point 5 is a multiple of five


def test_a_page_that_reloads_gets_the_whole_history_again(
    stream_client: TestClient, gate: Gate
) -> None:
    with stream_client.websocket_connect(STREAM) as first:
        assert read_history(first)["index"] == [1, 2, 3]
        gate.open.set()
        while read_history(first)["index"] != [4, 5]:
            pass
    with stream_client.websocket_connect(STREAM) as second:
        reload = read_history(second)
    assert reload["reset"] is True
    assert reload["index"] == [1, 2, 3, 4, 5]


# --- The reset -------------------------------------------------------------------------------


def post_reset(client: TestClient, **kwargs: Any) -> Any:
    return client.post(f"{API}/alignment/focus/reset", **kwargs)


def test_a_reset_with_the_token_restarts_the_best_value(
    client: TestClient, core: FakeCoreClient
) -> None:
    response = post_reset(client, headers=bearer())
    assert response.status_code == 200
    assert response.json() == {"reset": True, "message": "the best focus value restarted"}
    assert core.focus_resets == 1
    post_reset(client, headers=bearer())
    assert core.focus_resets == 2
    assert core.submitted == []  # it is no scheduler command


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token"}])
def test_a_reset_without_the_right_token_is_refused(
    client: TestClient, core: FakeCoreClient, headers: dict[str, str]
) -> None:
    response = post_reset(client, headers=headers)
    assert response.status_code == 401
    assert core.focus_resets == 0


def test_a_server_without_a_token_hash_refuses_the_reset(
    make_app: Callable[..., FastAPI],
    open_client: Callable[..., TestClient],
    seeded: Store,
    core: FakeCoreClient,
) -> None:
    client = open_client(make_app(token_hash=None))
    response = post_reset(client, headers=bearer())
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "commands_disabled"
    assert core.focus_resets == 0


def test_the_reset_counts_against_the_limit_of_commands(
    client: TestClient, core: FakeCoreClient, clock: VirtualClock
) -> None:
    for _ in range(5):  # the limit of the test settings is 5 for each 60 s
        assert post_reset(client, headers=bearer()).status_code == 200
    response = post_reset(client, headers=bearer())
    assert response.status_code == 429
    assert core.focus_resets == 5
    clock.advance(61)
    assert post_reset(client, headers=bearer()).status_code == 200


def test_a_reset_while_core_is_down_is_a_503(client: TestClient, core: FakeCoreClient) -> None:
    core.fail_with = CoreUnavailableError("gone")
    response = post_reset(client, headers=bearer())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "core_unavailable"


def test_the_token_never_comes_back_in_the_answer(client: TestClient) -> None:
    response = post_reset(client, headers=bearer())
    assert TOKEN not in response.text


def test_the_state_of_the_http_route_holds_the_whole_history(
    client: TestClient, core: FakeCoreClient
) -> None:
    from seeingmon.scheduler.commands import StartAlignment

    core.submit(StartAlignment())
    core.set_alignment_state(
        AlignmentState(active=True, focus=FocusView(fwhm_px=2.5, history=history(range(1, 9))))
    )
    body = client.get(f"{API}/alignment/state").json()
    assert body["focus"]["history"]["reset"] is True
    assert body["focus"]["history"]["index"] == list(range(1, 9))
