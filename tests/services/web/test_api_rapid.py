"""The rapid focus routes: start and stop, the offer in the state, and the readings on the video.

The start route sends no position: `core` knows where Polaris is. The stop route is a plain
scheduler command. The readings travel in the state of the video of Polaris, `core` sends all of
them in each frame, and each WebSocket viewer gets only the ones that it lacks.
"""

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
from seeingmon.scheduler.commands import (
    CommandResult,
    RejectReason,
    StartAlignment,
    StartRapidFocus,
    StopRapidFocus,
)
from seeingmon.services.web.contract import (
    AlignmentState,
    PolarisFrame,
    RapidFocusView,
    RapidReadingsView,
)
from seeingmon.services.web.core_client import (
    CoreProtocolError,
    CoreUnavailableError,
    FakeCoreClient,
)
from seeingmon.services.web.live import RAPID_LISTS, RAPID_PATH, HistoryCursor
from seeingmon.store.db import Store
from tests.services.web.client import TestClient
from tests.services.web.helpers import TOKEN, bearer, polaris_state, tiny_png

API = "/api/v1"
START = f"{API}/alignment/rapid-focus/start"
STOP = f"{API}/alignment/rapid-focus/stop"
STREAM = f"{API}/polaris/stream"
FRAME = f"{API}/polaris/frame"
TOO_WIDE = "the stars are too wide for the rapid mode: 21 arcsec, the limit is 12"


def readings(indexes: range, session: int = 1) -> RapidReadingsView:
    """The readings `indexes`, one every 50 ms, whose width falls from 4 arcseconds."""
    points = list(indexes)
    return RapidReadingsView(
        session=session,
        reset=True,
        index=points,
        t_utc_ms=[1_800_000_000_000 + 50 * n for n in points],
        fwhm_arcsec=[round(4.0 - 0.01 * n, 3) for n in points],
        peak_fraction=[0.3] * len(points),
        n_frames=[4] * len(points),
        spike=[n % 5 == 0 for n in points],
        saturated=[False] * len(points),
    )


def running(indexes: range, session: int = 1) -> RapidFocusView:
    """What the mode shows while it runs, with the readings `indexes`."""
    return RapidFocusView(
        available=True,
        active=True,
        since_utc="2026-10-05T21:00:00.000Z",
        mode="bin1",
        exposure_us=2000,
        gain=0,
        n_stars=1,
        fwhm_arcsec=3.9,
        best_fwhm_arcsec=3.8,
        peak_fraction=0.3,
        readings=readings(indexes, session),
    )


def state_with(indexes: range, session: int = 1) -> dict[str, Any]:
    """The JSON of a Polaris state whose rapid focus view holds the readings `indexes`."""
    state = polaris_state(live=False).model_copy(update={"rapid_focus": running(indexes, session)})
    return state.model_dump(mode="json")


def post(client: TestClient, path: str, body: Any = None, **kwargs: Any) -> Any:
    return client.post(path, json=body, headers=bearer(), **kwargs)


@pytest.fixture
def aligning(core: FakeCoreClient) -> FakeCoreClient:
    """A fake core whose scheduler aligns, with the rapid focus mode on offer."""
    assert core.submit(StartAlignment()).accepted
    core.submitted.clear()
    return core


# --- Start -----------------------------------------------------------------------------------


class TestStart:
    def test_a_start_without_a_body_asks_core_to_choose_the_center(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        response = post(client, START)
        assert response.status_code == 200
        assert response.json() == {
            "accepted": True,
            "message": "rapid focus started",
            "state": "align",
            "reason": None,
            "task_id": None,
        }
        assert aligning.rapid_starts == [(None, None)]
        assert aligning.submitted == [StartRapidFocus(*aligning.rapid_center)]

    def test_an_empty_object_is_the_same_as_no_body(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        assert post(client, START, {}).status_code == 200
        assert aligning.rapid_starts == [(None, None)]

    def test_the_exposure_and_the_gain_reach_core(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        response = post(client, START, {"exposure_us": 1500, "gain": 40})
        assert response.status_code == 200
        assert aligning.rapid_starts == [(1500, 40)]
        assert aligning.submitted == [StartRapidFocus(*aligning.rapid_center, 1500, 40)]

    def test_a_start_while_the_mode_runs_keeps_it_alive(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        assert post(client, START).status_code == 200
        aligning.rapid_offer = TOO_WIDE  # the offer lapses, and a running mode is not affected
        again = post(client, START)
        assert again.status_code == 200
        assert again.json()["message"] == "rapid focus already runs, so the idle timer restarted"

    def test_the_request_carries_no_position(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        for body in ({"center_x_px": 1.0, "center_y_px": 2.0}, {"x": 1}, {"roi": [1, 2, 3, 4]}):
            response = post(client, START, body)
            assert response.status_code == 422, body
            assert response.json()["error"]["code"] == "invalid_request"
        assert aligning.rapid_starts == []

    @pytest.mark.parametrize(
        "body",
        [
            {"exposure_us": 0},
            {"exposure_us": -5},
            {"exposure_us": 1.5},
            {"exposure_us": "fast"},
            {"exposure_us": True},
            {"exposure_us": 2_000_000_001},
            {"gain": -1},
            {"gain": 1001},
            {"gain": 2.5},
            {"gain": "high"},
            [1, 2],
        ],
    )
    def test_a_body_outside_the_bounds_never_reaches_core(
        self, client: TestClient, aligning: FakeCoreClient, body: Any
    ) -> None:
        response = post(client, START, body)
        assert response.status_code == 422
        assert "Traceback" not in response.text
        assert aligning.rapid_starts == []

    def test_outside_the_alignment_the_answer_is_409_not_aligning(
        self, client: TestClient, core: FakeCoreClient
    ) -> None:
        response = post(client, START)
        assert response.status_code == 409
        assert response.json() == {
            "accepted": False,
            "message": "no alignment runs",
            "state": "auto",
            "reason": "not_aligning",
            "task_id": None,
        }

    def test_a_mode_that_core_does_not_offer_says_what_is_missing(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        aligning.rapid_offer = TOO_WIDE
        response = post(client, START)
        assert response.status_code == 409
        body = response.json()
        assert (body["accepted"], body["reason"]) == (False, "not_available")
        assert body["message"] == TOO_WIDE
        assert aligning.submitted == []

    @pytest.mark.parametrize(
        ("reason", "status"),
        [
            (RejectReason.NOT_ALIGNING, 409),
            (RejectReason.NOT_AVAILABLE, 409),
            (RejectReason.CAMERA_FAULT, 409),
            (RejectReason.DEGRADED, 409),
            (RejectReason.INVALID, 422),
            (RejectReason.CLOSED, 503),
        ],
    )
    def test_every_answer_of_the_scheduler_has_a_status(
        self, client: TestClient, aligning: FakeCoreClient, reason: RejectReason, status: int
    ) -> None:
        aligning.rapid_focus_start = lambda exposure_us=None, gain=None: CommandResult(  # type: ignore[method-assign]
            accepted=False,
            message=f"the scheduler said {reason.value}",
            state="align",
            reason=reason,
        )
        response = post(client, START, {"exposure_us": 1000})
        assert response.status_code == status
        if status == 409:
            assert response.json()["reason"] == reason.value
        else:
            assert response.json()["error"]["message"] == f"the scheduler said {reason.value}"

    def test_the_message_of_a_refusal_loses_its_paths(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        aligning.rapid_offer = "no solution in /home/someone/frames"  # repo-check: allow
        response = post(client, START)
        assert response.status_code == 409
        assert "someone" not in response.text

    def test_a_core_that_does_not_answer_is_a_503_with_a_retry_time(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        aligning.fail_with = CoreUnavailableError("core does not answer")
        response = post(client, START)
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
        assert response.json()["error"]["code"] == "core_unavailable"

    def test_a_core_of_an_older_version_is_a_502(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        aligning.fail_with = CoreProtocolError("core does not serve rapid_focus_start")
        response = post(client, START)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "core_error"


# --- Stop ------------------------------------------------------------------------------------


class TestStop:
    def test_a_stop_ends_the_mode_and_keeps_the_alignment(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        assert post(client, START).status_code == 200
        response = post(client, STOP)
        assert response.status_code == 200
        assert response.json()["message"] == "rapid focus stopped"
        assert response.json()["state"] == "align"
        assert aligning.rapid_running is False
        assert aligning.submitted[-1] == StopRapidFocus()

    def test_a_stop_while_the_mode_does_not_run_changes_nothing(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        response = post(client, STOP)
        assert response.status_code == 200
        assert response.json()["message"] == "rapid focus does not run, so nothing stops"

    def test_outside_the_alignment_the_answer_is_409_not_aligning(
        self, client: TestClient, core: FakeCoreClient
    ) -> None:
        response = post(client, STOP)
        assert response.status_code == 409
        assert response.json()["reason"] == "not_aligning"

    def test_a_stop_takes_no_body(self, client: TestClient, aligning: FakeCoreClient) -> None:
        assert post(client, STOP, {}).status_code == 200
        assert aligning.submitted == [StopRapidFocus()]

    def test_the_end_of_the_alignment_ends_the_mode_in_the_fake_too(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        assert post(client, START).status_code == 200
        assert aligning.rapid_running is True
        assert post(client, f"{API}/alignment/stop").status_code == 200
        assert aligning.rapid_running is False


# --- The token and the limits ----------------------------------------------------------------


@pytest.mark.parametrize("path", [START, STOP])
class TestTheToken:
    def test_a_command_without_the_token_never_reaches_core(
        self, client: TestClient, aligning: FakeCoreClient, path: str
    ) -> None:
        response = client.post(path, json=None)
        assert response.status_code == 401
        assert aligning.rapid_starts == []
        assert aligning.submitted == []

    def test_a_wrong_token_is_refused(
        self, client: TestClient, aligning: FakeCoreClient, path: str
    ) -> None:
        response = client.post(path, json=None, headers=bearer("wrong-token"))
        assert response.status_code == 401
        assert aligning.submitted == []

    def test_a_server_without_a_token_hash_refuses_the_command(
        self,
        make_app: Callable[..., FastAPI],
        open_client: Callable[..., TestClient],
        seeded: Store,
        aligning: FakeCoreClient,
        path: str,
    ) -> None:
        client = open_client(make_app(token_hash=None))
        response = client.post(path, json=None, headers=bearer())
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "commands_disabled"
        assert aligning.submitted == []

    def test_the_command_counts_against_the_limit_of_commands(
        self, client: TestClient, aligning: FakeCoreClient, clock: VirtualClock, path: str
    ) -> None:
        for _ in range(5):  # the limit of the test settings is 5 for each 60 s
            assert client.post(path, json=None, headers=bearer()).status_code == 200
        assert client.post(path, json=None, headers=bearer()).status_code == 429
        clock.advance(61)
        assert client.post(path, json=None, headers=bearer()).status_code == 200

    def test_the_token_never_comes_back_in_the_answer(
        self, client: TestClient, aligning: FakeCoreClient, path: str
    ) -> None:
        response = client.post(path, json=None, headers=bearer())
        assert TOKEN not in response.text


# --- The state -------------------------------------------------------------------------------


class TestTheState:
    def test_the_state_of_the_http_route_holds_the_whole_readings(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        aligning.set_alignment_state(AlignmentState(active=True, rapid_focus=running(range(1, 9))))
        body = client.get(f"{API}/alignment/state").json()
        assert body["rapid_focus"]["active"] is True
        assert body["rapid_focus"]["readings"]["reset"] is True
        assert body["rapid_focus"]["readings"]["index"] == list(range(1, 9))

    def test_the_offer_shows_in_the_state_of_the_http_route(
        self, client: TestClient, aligning: FakeCoreClient
    ) -> None:
        offer = RapidFocusView(
            available=False, reason=TOO_WIDE, coarse_fwhm_arcsec=21.0, max_fwhm_arcsec=12.0
        )
        aligning.set_alignment_state(AlignmentState(active=True, rapid_focus=offer))
        body = client.get(f"{API}/alignment/state").json()["rapid_focus"]
        assert (body["available"], body["reason"]) == (False, TOO_WIDE)
        assert (body["coarse_fwhm_arcsec"], body["max_fwhm_arcsec"]) == (21.0, 12.0)
        assert body["readings"] is None


# --- The readings of a viewer ----------------------------------------------------------------


class TestTheCursorOfAViewer:
    def cursor(self) -> HistoryCursor:
        return HistoryCursor(RAPID_PATH, RAPID_LISTS)

    def test_the_lists_are_the_lists_of_the_readings_of_the_contract(self) -> None:
        fields = set(RapidReadingsView.model_fields) - {"session", "reset"}
        assert set(RAPID_LISTS) == fields
        assert RAPID_PATH == ("rapid_focus", "readings")

    def test_the_first_message_holds_all_the_readings_and_says_reset(self) -> None:
        state = state_with(range(1, 6))
        self.cursor().delta(state)
        assert state["rapid_focus"]["readings"]["reset"] is True
        assert state["rapid_focus"]["readings"]["index"] == [1, 2, 3, 4, 5]

    def test_the_next_message_holds_only_the_new_readings_of_every_list(self) -> None:
        cursor = self.cursor()
        cursor.delta(state_with(range(1, 6)))
        state = state_with(range(3, 9))  # the window of core moved on
        cursor.delta(state)
        view = state["rapid_focus"]["readings"]
        assert view["reset"] is False
        assert view["index"] == [6, 7, 8]
        for name in RAPID_LISTS:
            assert len(view[name]) == 3, name
        assert view["fwhm_arcsec"] == [3.94, 3.93, 3.92]
        assert view["spike"] == [False, False, False]

    def test_a_viewer_that_skipped_messages_gets_every_reading_that_it_lacks(self) -> None:
        cursor = self.cursor()
        cursor.delta(state_with(range(1, 4)))
        state = state_with(range(1, 40))
        cursor.delta(state)
        assert state["rapid_focus"]["readings"]["index"] == list(range(4, 40))

    def test_a_new_session_resets_the_viewer(self) -> None:
        cursor = self.cursor()
        cursor.delta(state_with(range(1, 100), session=1))
        state = state_with(range(1, 4), session=2)
        cursor.delta(state)
        assert state["rapid_focus"]["readings"]["reset"] is True
        assert state["rapid_focus"]["readings"]["index"] == [1, 2, 3]

    def test_readings_that_run_backward_reset_the_viewer(self) -> None:
        cursor = self.cursor()
        cursor.delta(state_with(range(1, 100), session=1))
        state = state_with(range(1, 4), session=1)  # core restarted: the same session number
        cursor.delta(state)
        assert state["rapid_focus"]["readings"]["reset"] is True

    def test_a_state_without_readings_stays_as_it_is(self) -> None:
        for state in (
            polaris_state().model_dump(mode="json"),
            polaris_state(live=False)
            .model_copy(update={"rapid_focus": RapidFocusView(available=True)})
            .model_dump(mode="json"),
        ):
            before = json.dumps(state)
            self.cursor().delta(state)
            assert json.dumps(state) == before

    def test_the_cursor_of_the_focus_history_still_ignores_these_readings(self) -> None:
        state = state_with(range(1, 6))
        HistoryCursor().delta(state)
        assert state["rapid_focus"]["readings"]["reset"] is True  # untouched
        assert len(state["rapid_focus"]["readings"]["index"]) == 5


class Gate:
    """Lets a test hold a source back until the viewer has read what came before."""

    def __init__(self) -> None:
        self.open = threading.Event()


def gated_frames(gate: Gate) -> Callable[[], AsyncIterator[PolarisFrame]]:
    """Frames of the mode with the readings 1 to 3, then (after the gate opens) 1 to 5."""

    def frame(last: int) -> PolarisFrame:
        state = polaris_state(seq=0, live=False).model_copy(
            update={"rapid_focus": running(range(1, last + 1))}
        )
        return PolarisFrame(state, tiny_png(last * 10, (128, 128)))

    async def source() -> AsyncIterator[PolarisFrame]:
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
    core = FakeCoreClient(clock=clock, polaris=gated_frames(gate))
    return open_client(make_app(core=core, sleep=short_sleep))


def read_readings(session: WebSocketTestSession) -> dict[str, Any]:
    message = session.receive_json()
    assert message["type"] == "state"
    session.receive_bytes()
    return message["state"]["rapid_focus"]["readings"]  # type: ignore[no-any-return]


class TestTheVideo:
    def test_a_viewer_gets_all_the_readings_first_and_then_only_the_new_ones(
        self, stream_client: TestClient, gate: Gate
    ) -> None:
        with stream_client.websocket_connect(STREAM) as session:
            first = read_readings(session)
            assert first["reset"] is True
            assert first["index"] == [1, 2, 3]
            assert first["fwhm_arcsec"] == [3.99, 3.98, 3.97]
            gate.open.set()
            later = read_readings(session)
            while later["index"] == []:  # a message without a new reading may come first
                later = read_readings(session)
            assert later["reset"] is False
            assert later["index"] == [4, 5]
            assert later["spike"] == [False, True]  # the reading 5 is a multiple of five
            assert later["n_frames"] == [4, 4]

    def test_a_page_that_reloads_gets_all_the_readings_again(
        self, stream_client: TestClient, gate: Gate
    ) -> None:
        with stream_client.websocket_connect(STREAM) as first:
            assert read_readings(first)["index"] == [1, 2, 3]
            gate.open.set()
            while read_readings(first)["index"] != [4, 5]:
                pass
        with stream_client.websocket_connect(STREAM) as second:
            reload = read_readings(second)
        assert reload["reset"] is True
        assert reload["index"] == [1, 2, 3, 4, 5]

    def test_two_viewers_keep_their_own_cursors(
        self, stream_client: TestClient, gate: Gate
    ) -> None:
        with (
            stream_client.websocket_connect(STREAM) as one,
            stream_client.websocket_connect(STREAM) as two,
        ):
            assert read_readings(one)["index"] == [1, 2, 3]
            assert read_readings(two)["index"] == [1, 2, 3]
            gate.open.set()
            for session in (one, two):
                later = read_readings(session)
                while later["index"] == []:
                    later = read_readings(session)
                assert later["index"] == [4, 5]

    def test_the_poll_route_keeps_the_readings_out_of_its_header(
        self, stream_client: TestClient
    ) -> None:
        response = stream_client.get(FRAME)  # the first call starts the stream
        for _ in range(100):
            if response.status_code == 200:
                break
            asyncio.run(asyncio.sleep(0.02))
            response = stream_client.get(FRAME)
        assert response.status_code == 200
        state = json.loads(response.headers["x-frame-state"])
        rapid = state["rapid_focus"]
        assert rapid["readings"] is None
        assert (rapid["active"], rapid["n_stars"]) == (True, 1)
        assert (rapid["fwhm_arcsec"], rapid["best_fwhm_arcsec"]) == (3.9, 3.8)
        assert len(response.headers["x-frame-state"]) < 2000  # a header, not a document

    def test_a_poll_of_a_video_without_the_mode_has_no_view(
        self,
        make_app: Callable[..., FastAPI],
        open_client: Callable[..., TestClient],
        seeded: Store,
        clock: VirtualClock,
    ) -> None:
        async def plain() -> AsyncIterator[PolarisFrame]:
            yield PolarisFrame(polaris_state(seq=0), tiny_png(10, (128, 128)))
            await asyncio.Event().wait()

        core = FakeCoreClient(clock=clock, polaris=plain)
        client = open_client(make_app(core=core, sleep=short_sleep))
        response = client.get(FRAME)
        for _ in range(100):
            if response.status_code == 200:
                break
            asyncio.run(asyncio.sleep(0.02))
            response = client.get(FRAME)
        assert response.status_code == 200
        assert json.loads(response.headers["x-frame-state"])["rapid_focus"] is None


# --- The description of the API --------------------------------------------------------------


class TestTheDescription:
    def test_the_two_operations_are_described(self, client: TestClient) -> None:
        document = client.get(f"{API}/openapi.json").json()
        start = document["paths"][START]["post"]
        stop = document["paths"][STOP]["post"]
        assert start["operationId"] == "post_alignment_rapid_focus_start"
        assert stop["operationId"] == "post_alignment_rapid_focus_stop"
        for operation in (start, stop):
            assert operation["tags"] == ["alignment"]
            assert operation["security"] == [{"bearerAuth": []}]
            assert {"200", "401", "403", "409", "422", "429", "502", "503"} <= set(
                operation["responses"]
            )

    def test_the_start_request_names_the_exposure_and_the_gain_and_no_position(
        self, client: TestClient
    ) -> None:
        document = client.get(f"{API}/openapi.json").json()
        schema = document["components"]["schemas"]["RapidFocusStartRequest"]
        assert set(schema["properties"]) == {"exposure_us", "gain"}
        assert schema["additionalProperties"] is False
        assert "required" not in schema

    def test_the_description_says_what_the_page_needs_to_know(self, client: TestClient) -> None:
        document = client.get(f"{API}/openapi.json").json()
        text = document["paths"][START]["post"]["description"]
        for phrase in (
            "rapid_focus.available",
            "rapid_focus.reason",
            "not_available",
            "idle timer",
            "carries no position",
        ):
            assert phrase in text, phrase

    def test_the_view_is_part_of_the_alignment_state_and_the_polaris_state(
        self, client: TestClient
    ) -> None:
        schemas = client.get(f"{API}/openapi.json").json()["components"]["schemas"]
        for name in ("AlignmentState", "PolarisState"):
            assert "rapid_focus" in schemas[name]["properties"], name
        assert {"RapidFocusView", "RapidReadingsView"} <= set(schemas)
