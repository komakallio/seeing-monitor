"""The start of the rapid focus mode through the clients of `core`: the fake and the RPC client.

`FakeCoreClient` follows the rules of the scheduler for the two commands and the offer of `core`.
`RpcCoreClient` sends the start as the method `rapid_focus_start`, with the exposure and the gain
and no position, and it carries the other two commands in `submit`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.scheduler.commands import (
    Pause,
    RejectReason,
    StartAlignment,
    StartRapidFocus,
    StopAlignment,
    StopRapidFocus,
)
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.web.contract import METHOD_RAPID_FOCUS_START
from seeingmon.services.web.core_client import (
    CoreProtocolError,
    CoreUnavailableError,
    FakeCoreClient,
    RpcCoreClient,
)
from tests.services.web.helpers import ReferenceCore

OFFER = "the stars are too wide for the rapid mode: 21 arcsec, the limit is 12"


# --- FakeCoreClient --------------------------------------------------------------------------


@pytest.fixture
def fake() -> FakeCoreClient:
    fake = FakeCoreClient(clock=VirtualClock(1_800_000_000_000_000_000))
    assert fake.submit(StartAlignment()).accepted
    fake.submitted.clear()
    return fake


class TestTheFake:
    def test_the_commands_need_an_alignment(self) -> None:
        fake = FakeCoreClient()
        for command in (StartRapidFocus(1.0, 2.0), StopRapidFocus()):
            result = fake.submit(command)
            assert (result.accepted, result.reason) == (False, RejectReason.NOT_ALIGNING)
        assert fake.rapid_running is False

    def test_a_start_runs_the_mode_and_a_second_start_keeps_it_alive(
        self, fake: FakeCoreClient
    ) -> None:
        first = fake.submit(StartRapidFocus(1.0, 2.0))
        second = fake.submit(StartRapidFocus(1.0, 2.0))
        assert (first.accepted, first.message) == (True, "rapid focus started")
        assert second.message == "rapid focus already runs, so the idle timer restarted"
        assert fake.rapid_running is True

    def test_a_stop_ends_it_and_a_second_stop_changes_nothing(self, fake: FakeCoreClient) -> None:
        fake.submit(StartRapidFocus(1.0, 2.0))
        assert fake.submit(StopRapidFocus()).message == "rapid focus stopped"
        again = fake.submit(StopRapidFocus())
        assert (again.accepted, again.message) == (
            True,
            "rapid focus does not run, so nothing stops",
        )

    @pytest.mark.parametrize("ending", [StopAlignment(), Pause()])
    def test_the_end_of_the_alignment_ends_the_mode(
        self, fake: FakeCoreClient, ending: StopAlignment | Pause
    ) -> None:
        fake.submit(StartRapidFocus(1.0, 2.0))
        assert fake.submit(ending).accepted
        assert fake.rapid_running is False

    def test_a_degraded_camera_refuses_the_start(self, fake: FakeCoreClient) -> None:
        fake.degraded = True
        result = fake.submit(StartRapidFocus(1.0, 2.0))
        assert (result.accepted, result.reason) == (False, RejectReason.DEGRADED)
        assert fake.rapid_running is False

    def test_the_start_method_uses_the_center_of_the_offer(self, fake: FakeCoreClient) -> None:
        fake.rapid_center = (800.5, 600.25)
        result = fake.rapid_focus_start(1500, 40)
        assert result.accepted
        assert fake.rapid_starts == [(1500, 40)]
        assert fake.submitted == [StartRapidFocus(800.5, 600.25, 1500, 40)]

    def test_a_mode_that_is_not_offered_is_refused_with_the_reason_in_words(
        self, fake: FakeCoreClient
    ) -> None:
        fake.rapid_offer = OFFER
        result = fake.rapid_focus_start()
        assert (result.accepted, result.reason, result.message) == (
            False,
            RejectReason.NOT_AVAILABLE,
            OFFER,
        )
        assert fake.submitted == []
        assert fake.rapid_running is False

    def test_a_running_mode_survives_a_lapsed_offer(self, fake: FakeCoreClient) -> None:
        assert fake.rapid_focus_start().accepted
        fake.rapid_offer = OFFER
        assert fake.rapid_focus_start().accepted

    def test_outside_the_alignment_the_start_method_says_so(self) -> None:
        fake = FakeCoreClient()
        result = fake.rapid_focus_start()
        assert (result.accepted, result.reason) == (False, RejectReason.NOT_ALIGNING)

    def test_a_failing_fake_raises_for_the_start_method(self, fake: FakeCoreClient) -> None:
        fake.fail_with = CoreUnavailableError("gone")
        with pytest.raises(CoreUnavailableError):
            fake.rapid_focus_start()
        assert fake.rapid_starts == []


# --- RpcCoreClient ---------------------------------------------------------------------------


@pytest.fixture
def clients() -> Iterator[list[RpcCoreClient]]:
    opened: list[RpcCoreClient] = []
    yield opened
    for client in opened:
        client.close()


@pytest.fixture
def start_core(
    endpoint: Endpoint, key: ConnectionKey, servers: list[IpcServer]
) -> Callable[..., ReferenceCore]:
    def start(prepare: Callable[[ReferenceCore], object] | None = None) -> ReferenceCore:
        core = ReferenceCore(endpoint, key)
        if prepare is not None:
            prepare(core)
        servers.append(core.start())
        return core

    return start


def connect(clients: list[RpcCoreClient], core: ReferenceCore, key: ConnectionKey) -> RpcCoreClient:
    client = RpcCoreClient(
        core.endpoint,
        key,
        connect_timeout_s=0.5,
        handshake_timeout_s=2.0,
        rpc_timeout_s=2.0,
        retry_interval_s=0.0,
    )
    clients.append(client)
    return client


class TestTheRpcClient:
    def test_the_start_reaches_core_as_a_method_with_the_two_settings(
        self,
        start_core: Callable[..., ReferenceCore],
        key: ConnectionKey,
        clients: list[RpcCoreClient],
    ) -> None:
        core = start_core(lambda c: c.backend.submit(StartAlignment()))
        client = connect(clients, core, key)
        result = client.rapid_focus_start(exposure_us=1500, gain=40)
        assert (result.accepted, result.message, result.state) == (
            True,
            "rapid focus started",
            "align",
        )
        assert core.backend.rapid_starts == [(1500, 40)]
        # The ReferenceCore chose the center, and the client sent none.
        assert core.backend.submitted[-1] == StartRapidFocus(*core.backend.rapid_center, 1500, 40)

    def test_a_start_without_settings_sends_nulls(
        self,
        start_core: Callable[..., ReferenceCore],
        key: ConnectionKey,
        clients: list[RpcCoreClient],
    ) -> None:
        core = start_core(lambda c: c.backend.submit(StartAlignment()))
        assert connect(clients, core, key).rapid_focus_start().accepted
        assert core.backend.rapid_starts == [(None, None)]

    def test_a_refusal_of_core_is_a_result_with_its_reason_and_message(
        self,
        start_core: Callable[..., ReferenceCore],
        key: ConnectionKey,
        clients: list[RpcCoreClient],
    ) -> None:
        def prepare(core: ReferenceCore) -> None:
            core.backend.submit(StartAlignment())
            core.backend.rapid_offer = OFFER

        core = start_core(prepare)
        result = connect(clients, core, key).rapid_focus_start()
        assert (result.accepted, result.reason, result.message) == (
            False,
            RejectReason.NOT_AVAILABLE,
            OFFER,
        )

    def test_outside_the_alignment_the_refusal_is_not_aligning(
        self,
        start_core: Callable[..., ReferenceCore],
        key: ConnectionKey,
        clients: list[RpcCoreClient],
    ) -> None:
        core = start_core()
        result = connect(clients, core, key).rapid_focus_start()
        assert (result.accepted, result.reason) == (False, RejectReason.NOT_ALIGNING)

    def test_the_other_two_commands_travel_in_submit(
        self,
        start_core: Callable[..., ReferenceCore],
        key: ConnectionKey,
        clients: list[RpcCoreClient],
    ) -> None:
        core = start_core(lambda c: c.backend.submit(StartAlignment()))
        client = connect(clients, core, key)
        assert client.submit(StartRapidFocus(1036.0, 705.5, exposure_us=2000, gain=0)).accepted
        assert client.submit(StopRapidFocus()).accepted
        assert core.backend.submitted[-2:] == [
            StartRapidFocus(1036.0, 705.5, exposure_us=2000, gain=0),
            StopRapidFocus(),
        ]

    def test_a_core_without_the_method_is_a_protocol_error(
        self,
        start_core: Callable[..., ReferenceCore],
        key: ConnectionKey,
        clients: list[RpcCoreClient],
    ) -> None:
        core = start_core(lambda c: c.handlers.pop(METHOD_RAPID_FOCUS_START))
        with pytest.raises(CoreProtocolError, match="the versions differ"):
            connect(clients, core, key).rapid_focus_start()

    @pytest.mark.parametrize(
        "answer",
        [
            None,
            [],
            "accepted",
            {"accepted": "yes", "message": "x", "state": "align"},
            {"accepted": True},
        ],
    )
    def test_an_unreadable_answer_is_a_protocol_error(
        self,
        start_core: Callable[..., ReferenceCore],
        key: ConnectionKey,
        clients: list[RpcCoreClient],
        answer: Any,
    ) -> None:
        def junk(params: Mapping[str, Any]) -> Any:
            return answer

        core = start_core(lambda c: c.handlers.__setitem__(METHOD_RAPID_FOCUS_START, junk))
        with pytest.raises(CoreProtocolError, match="unreadable command result"):
            connect(clients, core, key).rapid_focus_start()

    def test_a_core_that_is_down_is_unavailable(
        self, endpoint: Endpoint, key: ConnectionKey, clients: list[RpcCoreClient]
    ) -> None:
        client = RpcCoreClient(endpoint, key, connect_timeout_s=0.05, retry_interval_s=0.0)
        clients.append(client)
        with pytest.raises(CoreUnavailableError):
            client.rapid_focus_start()

    def test_the_start_is_not_a_scheduler_command_that_the_client_could_build(self) -> None:
        # The center belongs to `core`: the client has no way to name one in `rapid_focus_start`.
        parameters = inspect.signature(RpcCoreClient.rapid_focus_start).parameters
        assert set(parameters) == {"self", "exposure_us", "gain"}
