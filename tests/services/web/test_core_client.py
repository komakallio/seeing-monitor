"""`FakeCoreClient`, and `RpcCoreClient` against a reference `core` on the real connection layer."""

from __future__ import annotations

import asyncio
import secrets
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.scheduler.commands import (
    Command,
    Pause,
    QueueBurst,
    QueueDark,
    QueueReplay,
    QueueSweep,
    RejectReason,
    Resume,
    StartAlignment,
    StopAlignment,
)
from seeingmon.services.config import ServicesConfig
from seeingmon.services.ipc.endpoint import Endpoint
from seeingmon.services.ipc.errors import IpcConnectError
from seeingmon.services.ipc.keys import ConnectionKey
from seeingmon.services.ipc.rpc import connect_rpc
from seeingmon.services.ipc.server import IpcServer
from seeingmon.services.web.config import CoreLinkSettings
from seeingmon.services.web.contract import (
    METHOD_DARK_LIBRARY,
    METHOD_PING,
    METHOD_STATUS,
    METHOD_SUBMIT,
    METHODS,
    AlignmentFrame,
    AlignmentState,
)
from seeingmon.services.web.core_client import (
    CoreClient,
    CoreProtocolError,
    CoreUnavailableError,
    FakeCoreClient,
    RpcCoreClient,
)
from seeingmon.services.web.fake_dark import DarkScript
from tests.services.conftest import wait_until
from tests.services.web.helpers import ReferenceCore, alignment_state, dark_set, tiny_jpeg


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


# --- FakeCoreClient --------------------------------------------------------------------------


def test_the_fake_and_the_real_client_both_satisfy_the_protocol(
    native: Endpoint, key: ConnectionKey
) -> None:
    assert isinstance(FakeCoreClient(), CoreClient)
    assert isinstance(RpcCoreClient(native, key), CoreClient)


def test_the_fake_follows_the_alignment_rules_of_the_scheduler() -> None:
    fake = FakeCoreClient(state="auto")
    assert fake.submit(StopAlignment()).reason is RejectReason.NOT_ALIGNING
    started = fake.submit(StartAlignment())
    assert started.accepted
    assert started.state == "align"
    again = fake.submit(StartAlignment())
    assert again.accepted
    assert "already runs" in again.message
    assert fake.submit(Pause()).accepted
    assert fake.state == "paused"
    assert fake.submit(StartAlignment()).reason is RejectReason.PAUSED
    assert fake.submit(Pause()).reason is RejectReason.ALREADY_PAUSED
    assert fake.submit(Resume()).accepted
    assert fake.state == "safe"
    assert fake.submit(Resume()).reason is RejectReason.NOT_PAUSED


def test_the_fake_refuses_alignment_while_the_camera_is_degraded() -> None:
    fake = FakeCoreClient()
    fake.degraded = True
    assert fake.submit(StartAlignment()).reason is RejectReason.DEGRADED


def test_the_fake_queues_tasks_until_its_queue_is_full() -> None:
    fake = FakeCoreClient(max_queued=2)
    first = fake.submit(QueueBurst())
    second = fake.submit(QueueSweep())
    third = fake.submit(QueueReplay())
    assert (first.task_id, second.task_id) == (1, 2)
    assert "burst is queued" in first.message
    assert third.reason is RejectReason.QUEUE_FULL
    assert fake.status().scheduler.queued_tasks == 2


def test_the_fake_keeps_the_commands_in_order() -> None:
    fake = FakeCoreClient()
    commands: list[Command] = [StartAlignment(gain=5), StopAlignment(), Pause()]
    for command in commands:
        fake.submit(command)
    assert fake.submitted == commands


def test_the_fake_counts_the_commands_in_its_status() -> None:
    fake = FakeCoreClient()
    fake.submit(StopAlignment())
    fake.submit(StartAlignment())
    counters = fake.status().scheduler.counters
    assert counters == {"commands_accepted": 1, "commands_rejected": 1}


def test_the_fake_takes_its_times_from_the_clock() -> None:
    clock = VirtualClock(1_767_225_600_000_000_000)
    fake = FakeCoreClient(clock=clock)
    clock.advance(60)
    fake.submit(Pause())
    status = fake.status().scheduler
    assert status.t_utc_ns == 1_767_225_660_000_000_000
    assert status.state_since_utc_ns == 1_767_225_660_000_000_000


def test_the_fake_fails_every_call_while_told_to() -> None:
    fake = FakeCoreClient()
    fake.fail_with = CoreUnavailableError("core does not answer")
    with pytest.raises(CoreUnavailableError):
        fake.status()
    with pytest.raises(CoreUnavailableError):
        fake.submit(Pause())
    with pytest.raises(CoreUnavailableError):
        fake.alignment_state()
    assert fake.submitted == []
    fake.fail_with = None
    assert fake.status().instance == "fake-core"


def test_the_fake_reports_an_inactive_alignment_outside_align() -> None:
    fake = FakeCoreClient(alignment_state=alignment_state())
    assert fake.alignment_state() == AlignmentState(active=False)
    fake.submit(StartAlignment())
    assert fake.alignment_state() == alignment_state()
    fake.set_alignment_state(None)
    assert fake.alignment_state() == AlignmentState(active=True)


def frames_from(items: list[AlignmentFrame]) -> Callable[[], AsyncIterator[AlignmentFrame]]:
    async def source() -> AsyncIterator[AlignmentFrame]:
        for item in items:
            yield item

    return source


async def take(frames: AsyncIterator[AlignmentFrame], count: int) -> list[AlignmentFrame]:
    taken: list[AlignmentFrame] = []
    async for frame in frames:
        taken.append(frame)
        if len(taken) == count:
            break
    return taken


def test_the_fake_streams_the_frames_of_its_source() -> None:
    items = [AlignmentFrame(alignment_state(seq), tiny_jpeg(seq * 10)) for seq in (1, 2, 3)]
    fake = FakeCoreClient(frames=frames_from(items))
    got = run(take(fake.alignment_frames(), 3))
    assert got == items
    assert fake.streams_opened == 1
    assert fake.streams_closed == 1


def test_the_fake_counts_a_stream_that_the_caller_abandons() -> None:
    items = [AlignmentFrame(alignment_state(seq), tiny_jpeg()) for seq in (1, 2, 3)]
    fake = FakeCoreClient(frames=frames_from(items))

    async def main() -> None:
        stream = fake.alignment_frames()
        assert len(await take(stream, 1)) == 1
        await stream.aclose()  # type: ignore[attr-defined]

    run(main())
    assert (fake.streams_opened, fake.streams_closed) == (1, 1)


def test_a_fake_without_frames_holds_the_stream_open_until_cancelled() -> None:
    fake = FakeCoreClient()

    async def main() -> None:
        task = asyncio.ensure_future(take(fake.alignment_frames(), 1))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(main())
    assert (fake.streams_opened, fake.streams_closed) == (1, 1)


def test_the_fake_can_fail_its_stream() -> None:
    fake = FakeCoreClient()
    fake.fail_with = CoreUnavailableError("gone")
    with pytest.raises(CoreUnavailableError):
        run(take(fake.alignment_frames(), 1))


# --- RpcCoreClient ---------------------------------------------------------------------------


@pytest.fixture
def clients() -> Iterator[list[RpcCoreClient]]:
    opened: list[RpcCoreClient] = []
    yield opened
    for client in opened:
        client.close()


def make_client(
    clients: list[RpcCoreClient], endpoint: Endpoint, key: ConnectionKey, **options: Any
) -> RpcCoreClient:
    settings: dict[str, Any] = {
        "connect_timeout_s": 0.5,
        "handshake_timeout_s": 2.0,
        "rpc_timeout_s": 2.0,
        "retry_interval_s": 0.0,
        "poll_s": 0.05,
    }
    settings.update(options)
    client = RpcCoreClient(endpoint, key, **settings)
    clients.append(client)
    return client


@pytest.fixture
def start_core(
    endpoint: Endpoint, key: ConnectionKey, servers: list[IpcServer]
) -> Callable[..., ReferenceCore]:
    """Start a reference core at the endpoint of the test. Its `endpoint` is the address to use."""

    def start(
        prepare: Callable[[ReferenceCore], object] | None = None, **options: Any
    ) -> ReferenceCore:
        core = ReferenceCore(endpoint, key, **options)
        if prepare is not None:
            prepare(core)
        servers.append(core.start())
        return core

    return start


def test_the_client_reads_the_status_from_core(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core()
    client = make_client(clients, core.endpoint, key)
    status = client.status()
    assert status.instance == "reference-core"
    assert status.scheduler.state == "auto"
    assert client.ping() == "reference-core"
    assert client.connections == 1


def test_the_client_submits_commands_and_returns_the_answers(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core()
    client = make_client(clients, core.endpoint, key)
    started = client.submit(StartAlignment(exposure_s=0.5, gain=100))
    assert started.accepted
    assert started.state == "align"
    assert "already runs" in client.submit(StartAlignment()).message
    assert client.submit(Pause()).accepted
    rejected = client.submit(Pause())
    assert not rejected.accepted
    assert rejected.reason is RejectReason.ALREADY_PAUSED
    queued = client.submit(QueueBurst(duration_s=5.0, label="x"))
    assert queued.accepted
    assert queued.task_id == 1
    assert core.backend.submitted == [
        StartAlignment(exposure_s=0.5, gain=100),
        StartAlignment(),
        Pause(),
        Pause(),
        QueueBurst(duration_s=5.0, label="x"),
    ]


def test_the_client_reads_the_alignment_state(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core()
    client = make_client(clients, core.endpoint, key)
    assert client.alignment_state().active is False
    client.submit(StartAlignment())
    core.backend.set_alignment_state(alignment_state(5))
    assert client.alignment_state() == alignment_state(5)


def test_the_client_reuses_one_connection_for_many_calls(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core()
    client = make_client(clients, core.endpoint, key)
    for _ in range(5):
        client.status()
    assert client.connections == 1


def test_many_threads_can_call_at_once(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core()
    client = make_client(clients, core.endpoint, key)
    errors: list[Exception] = []

    def work() -> None:
        try:
            for _ in range(10):
                client.status()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert client.connections == 1


def test_the_client_reads_the_dark_library_and_queues_a_dark_task(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    backend = FakeCoreClient(instance="reference-core", dark_script=DarkScript(queued_s=3600.0))
    backend.dark.sets = [dark_set("set-a", 12.0, 5.0)]
    core = start_core(backend=backend)
    client = make_client(clients, core.endpoint, key)
    library = client.dark_library()
    assert [item.name for item in library.sets] == ["set-a"]
    assert library.task.state == "idle"
    command = QueueDark(
        exposure_s=20.0,
        frames=5,
        bias_frames=4,
        wait_for_cover=False,
        pause_after=False,
        label="by hand",
    )
    queued = client.submit(command)
    assert queued.accepted
    assert queued.task_id == 1
    assert core.backend.submitted == [command]  # the command crossed the wire unchanged
    task = client.dark_library().task
    assert (task.state, task.task_id) == ("queued", 1)
    assert (task.exposure_s, task.frames, task.bias_frames) == (20.0, 5, 4)
    assert (task.wait_for_cover, task.pause_after) == (False, False)
    busy = client.submit(QueueDark())
    assert not busy.accepted
    assert busy.reason is RejectReason.BUSY


def test_a_core_without_the_dark_method_is_a_protocol_error(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core(lambda core: core.handlers.pop(METHOD_DARK_LIBRARY))
    client = make_client(clients, core.endpoint, key)
    with pytest.raises(CoreProtocolError, match="versions differ"):
        client.dark_library()


def test_a_core_that_is_not_running_is_unavailable(
    endpoint: Endpoint, key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    client = make_client(clients, endpoint, key, connect_timeout_s=0.1)
    with pytest.raises(CoreUnavailableError, match="does not answer"):
        client.status()
    with pytest.raises(CoreUnavailableError):
        client.submit(Pause())


def test_a_failed_connection_is_not_retried_before_the_interval_passes(
    native: Endpoint,
    key: ConnectionKey,
    clients: list[RpcCoreClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    real = connect_rpc

    def counting(*args: Any, **kwargs: Any) -> Any:
        attempts.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr("seeingmon.services.web.core_client.connect_rpc", counting)
    clock = VirtualClock()
    client = make_client(
        clients, native, key, connect_timeout_s=0.05, retry_interval_s=5.0, clock=clock
    )
    for _ in range(3):
        with pytest.raises(CoreUnavailableError):
            client.status()
    assert len(attempts) == 1  # the next two calls failed at once, without a connection attempt
    clock.advance(5)
    with pytest.raises(CoreUnavailableError):
        client.status()
    assert len(attempts) == 2


def test_a_retry_after_a_failure_waits_for_core_only_a_short_time(
    native: Endpoint,
    key: ConnectionKey,
    servers: list[IpcServer],
    clients: list[RpcCoreClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first connection waits for the whole connect timeout, because core may be starting. A
    page that polls a core that is down must not wait that long on every refresh."""
    waits: list[float] = []
    down = [True]
    real = connect_rpc

    def recording(*args: Any, **kwargs: Any) -> Any:
        waits.append(kwargs["connect_timeout_s"])
        if down[0]:
            raise IpcConnectError("nothing answers")
        return real(*args, **kwargs)

    monkeypatch.setattr("seeingmon.services.web.core_client.connect_rpc", recording)
    clock = VirtualClock()
    client = make_client(
        clients,
        native,
        key,
        connect_timeout_s=2.0,
        retry_interval_s=3.0,
        probe_timeout_s=0.25,
        clock=clock,
    )
    for _ in range(
        3
    ):  # the first call waits for the whole timeout, and the next two answer at once
        with pytest.raises(CoreUnavailableError, match="does not answer"):
            client.status()
    assert waits == [2.0]
    clock.advance(2.9)  # still inside the window
    with pytest.raises(CoreUnavailableError):
        client.status()
    assert waits == [2.0]
    clock.advance(0.2)  # the window is over: one probe, and it is short
    with pytest.raises(CoreUnavailableError):
        client.status()
    assert waits == [2.0, 0.25]
    for _ in range(3):  # a long outage costs one short probe at the end of each window
        clock.advance(3.1)
        with pytest.raises(CoreUnavailableError):
            client.status()
    assert waits == [2.0, 0.25, 0.25, 0.25, 0.25]
    servers.append(ReferenceCore(native, key).start())  # core comes back
    down[0] = False
    clock.advance(3.1)
    assert client.status().instance == "reference-core"
    assert waits[-1] == 0.25
    # A link that worked and then broke starts over with the whole timeout.
    down[0] = True
    client.close()
    with pytest.raises(CoreUnavailableError):
        client.status()
    assert waits[-1] == 2.0


def test_the_probe_never_waits_longer_than_the_connect_timeout(
    native: Endpoint,
    key: ConnectionKey,
    clients: list[RpcCoreClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        waits.append(kwargs["connect_timeout_s"])
        raise IpcConnectError("nothing answers")

    monkeypatch.setattr("seeingmon.services.web.core_client.connect_rpc", recording)
    clock = VirtualClock()
    client = make_client(
        clients,
        native,
        key,
        connect_timeout_s=0.1,
        retry_interval_s=1.0,
        probe_timeout_s=5.0,
        clock=clock,
    )
    for _ in range(2):
        with pytest.raises(CoreUnavailableError):
            client.status()
        clock.advance(1.5)
    assert waits == [0.1, 0.1]


def test_the_link_settings_default_to_a_short_probe_and_a_few_seconds_of_memory() -> None:
    settings = CoreLinkSettings()
    assert settings.probe_timeout_s < settings.connect_timeout_s
    assert 2.0 <= settings.retry_interval_s <= 10.0


def test_the_client_takes_the_probe_timeout_from_the_configuration(native: Endpoint) -> None:
    services = ServicesConfig(connection_key=secrets.token_urlsafe(24), core_address=str(native))
    link = CoreLinkSettings(probe_timeout_s=0.5, retry_interval_s=4.0)
    client = RpcCoreClient.from_config(services, link, env={})
    assert client._probe_timeout_s == 0.5
    assert client._retry_interval_ns == 4_000_000_000


def test_the_client_connects_when_core_starts_later(
    native: Endpoint, key: ConnectionKey, servers: list[IpcServer], clients: list[RpcCoreClient]
) -> None:
    client = make_client(clients, native, key, connect_timeout_s=0.05)
    with pytest.raises(CoreUnavailableError):
        client.status()
    core = ReferenceCore(native, key)
    servers.append(core.start())
    assert client.status().scheduler.state == "auto"


def test_the_client_reconnects_after_core_restarts(
    native: Endpoint, key: ConnectionKey, servers: list[IpcServer], clients: list[RpcCoreClient]
) -> None:
    first = ReferenceCore(native, key)
    first.start()
    client = make_client(clients, native, key, connect_timeout_s=0.2)
    assert client.status().instance == "reference-core"
    first.stop()
    assert wait_until(lambda: client._rpc is not None and client._rpc.closed)
    with pytest.raises(CoreUnavailableError):
        client.status()
    second = ReferenceCore(native, key)
    servers.append(second.start())
    assert client.status().instance == "reference-core"
    assert client.connections == 2


def test_a_core_with_another_key_is_refused(
    start_core: Callable[..., ReferenceCore],
    other_key: ConnectionKey,
    clients: list[RpcCoreClient],
) -> None:
    core = start_core()
    client = make_client(clients, core.endpoint, other_key)
    with pytest.raises(CoreUnavailableError, match="refused the connection key"):
        client.status()


def test_a_method_that_core_lacks_is_a_protocol_error(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core(lambda core: core.handlers.pop(METHOD_STATUS))
    client = make_client(clients, core.endpoint, key)
    with pytest.raises(CoreProtocolError, match="versions differ"):
        client.status()


def test_an_error_inside_core_never_reaches_the_caller_as_text(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    def broken(params: Mapping[str, Any]) -> Any:
        raise RuntimeError("cannot open the private-folder-name file")

    core = start_core(lambda core: core.handlers.update({METHOD_STATUS: broken}))
    client = make_client(clients, core.endpoint, key)
    with pytest.raises(CoreProtocolError) as raised:
        client.status()
    assert "private-folder-name" not in str(raised.value)


def test_a_command_that_core_cannot_read_is_a_protocol_error(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core()
    client = make_client(clients, core.endpoint, key)
    # The reference core refuses a command that is not an object, like a core of another version.
    with pytest.raises(CoreProtocolError):
        client._call(METHOD_SUBMIT, {"command": "pause"}, 2.0)


UNREADABLE = {
    "ping": {"instance": 5},
    "status": {"nonsense": True},
    "submit": {"accepted": "yes"},
    "alignment_state": {"active": "yes"},
    "dark_library": {"mode": 5},
}


@pytest.mark.parametrize("method", sorted(UNREADABLE))
def test_an_unreadable_answer_is_a_protocol_error(
    start_core: Callable[..., ReferenceCore],
    key: ConnectionKey,
    clients: list[RpcCoreClient],
    method: str,
) -> None:
    core = start_core(
        lambda core: core.handlers.update({method: lambda params: UNREADABLE[method]})
    )
    client = make_client(clients, core.endpoint, key)
    calls: dict[str, Callable[[], object]] = {
        "ping": client.ping,
        "status": client.status,
        "submit": lambda: client.submit(Pause()),
        "alignment_state": client.alignment_state,
        "dark_library": client.dark_library,
    }
    with pytest.raises(CoreProtocolError):
        calls[method]()


def test_a_call_that_core_does_not_answer_in_time_is_unavailable_and_the_link_survives(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    release = threading.Event()

    def slow(params: Mapping[str, Any]) -> Any:
        release.wait(10)
        return {"instance": "late"}

    core = start_core(lambda core: core.handlers.update({METHOD_PING: slow}))
    client = make_client(clients, core.endpoint, key, rpc_timeout_s=0.3)
    with pytest.raises(CoreUnavailableError, match="did not answer ping in time"):
        client.ping()
    release.set()
    assert client.status().instance == "reference-core"  # the connection stayed usable
    assert client.connections == 1


def test_the_client_builds_itself_from_the_configuration(
    native: Endpoint, servers: list[IpcServer], clients: list[RpcCoreClient]
) -> None:
    text = secrets.token_urlsafe(24)
    core = ReferenceCore(native, ConnectionKey.from_text(text))
    servers.append(core.start())
    services = ServicesConfig(connection_key=text, core_address=str(native))
    client = RpcCoreClient.from_config(services, CoreLinkSettings(), env={})
    clients.append(client)
    assert client.status().instance == "reference-core"


# --- The alignment stream --------------------------------------------------------------------


def test_the_client_streams_the_frames_of_core(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    frames = [(alignment_state(seq), tiny_jpeg(seq * 20)) for seq in (1, 2, 3)]
    core = start_core(frames=frames)
    client = make_client(clients, core.endpoint, key)
    got = run(take(client.alignment_frames(), 3))
    assert [(item.state, item.jpeg) for item in got] == frames
    assert wait_until(lambda: core.stream_senders[0].closed)  # leaving the loop closed the stream


def test_a_client_that_stops_early_closes_the_stream(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    frames = [(alignment_state(seq), tiny_jpeg()) for seq in range(1, 6)]
    core = start_core(frames=frames)
    client = make_client(clients, core.endpoint, key)

    async def main() -> None:
        stream = client.alignment_frames()
        assert len(await take(stream, 2)) == 2
        await stream.aclose()  # type: ignore[attr-defined]

    run(main())
    assert wait_until(lambda: core.stream_senders[0].closed)


def test_a_stream_that_core_ends_raises_unavailable_after_the_last_frame(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core(frames=[(alignment_state(1), tiny_jpeg())])
    client = make_client(clients, core.endpoint, key)

    async def main() -> None:
        stream = client.alignment_frames()
        first = await stream.__anext__()
        assert first.state == alignment_state(1)
        assert await asyncio.to_thread(core.sending_done.wait, 5)
        core.end_streams()
        with pytest.raises(CoreUnavailableError, match="stream of core closed"):
            await stream.__anext__()

    run(main())


def test_a_message_that_is_not_a_frame_is_a_protocol_error(
    start_core: Callable[..., ReferenceCore], key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core(lambda core: core.raw_payloads.append(b"this is not an alignment frame"))
    client = make_client(clients, core.endpoint, key)
    with pytest.raises(CoreProtocolError):
        run(take(client.alignment_frames(), 1))


def test_the_stream_of_a_core_that_is_down_is_unavailable(
    endpoint: Endpoint, key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    client = make_client(clients, endpoint, key, connect_timeout_s=0.05)
    with pytest.raises(CoreUnavailableError):
        run(take(client.alignment_frames(), 1))


def test_a_stream_with_the_wrong_key_is_unavailable(
    start_core: Callable[..., ReferenceCore], other_key: ConnectionKey, clients: list[RpcCoreClient]
) -> None:
    core = start_core()
    client = make_client(clients, core.endpoint, other_key)
    with pytest.raises(CoreUnavailableError, match="refused"):
        run(take(client.alignment_frames(), 1))


def test_the_methods_of_the_reference_core_match_the_contract(
    start_core: Callable[..., ReferenceCore],
) -> None:
    assert set(start_core().handlers) == set(METHODS)
