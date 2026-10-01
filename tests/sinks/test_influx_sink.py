"""The InfluxDB sink against a local HTTP server that simulates outages and errors."""

from __future__ import annotations

import base64
import random
import ssl
import time
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.sinks.base import Sink, SinkError, StoredRow
from seeingmon.sinks.config import InfluxSinkConfig
from seeingmon.sinks.forwarder import Forwarder
from seeingmon.sinks.influx import InfluxSink, make_opener
from seeingmon.store.config import ForwarderConfig
from seeingmon.store.db import Store
from tests.sinks.fake_influx import Drop, FakeInfluxServer, Hang, Reply
from tests.sinks.line_protocol import parse_line, split_lines
from tests.store.builders import NS_PER_S, T0, make_health, make_window

TOKEN = "example-token-value"
USER = "example-user"
CODE = "example-code-value"  # the password of a version 1 server


@pytest.fixture
def server() -> Iterator[FakeInfluxServer]:
    with FakeInfluxServer() as running:
        yield running


def config(server: FakeInfluxServer, **overrides: Any) -> InfluxSinkConfig:
    values: dict[str, Any] = {
        "kind": "influx",
        "endpoint": server.url,
        "org": "example-org",
        "bucket": "seeing",
        "timeout_s": 5,
    }
    values.update(overrides)
    return InfluxSinkConfig(**values)


def make_sink(server: FakeInfluxServer, **overrides: Any) -> InfluxSink:
    settings = config(server, **overrides)
    secret = TOKEN if settings.version == 2 else None
    return InfluxSink(
        "influx",
        settings,
        token=secret,
        password=CODE if settings.version == 1 else None,
        opener=make_opener(use_environment_proxies=False),
    )


def health_rows(count: int = 3, start: int = 0) -> list[StoredRow]:
    return [
        StoredRow(n + 1, make_health(T0 + n * NS_PER_S).to_row())
        for n in range(start, start + count)
    ]


class TestRequests:
    def test_a_version_2_write_posts_line_protocol_with_the_token_header(
        self, server: FakeInfluxServer
    ) -> None:
        make_sink(server).send("health", health_rows(3))
        (request,) = server.requests
        assert request.method == "POST"
        assert request.path == "/api/v2/write"
        assert request.query == {
            "org": ["example-org"],
            "bucket": ["seeing"],
            "precision": ["ns"],
        }
        assert request.headers["authorization"] == f"Token {TOKEN}"
        assert request.headers["content-type"].startswith("text/plain")
        lines = split_lines(request.body)
        assert len(lines) == 3
        points = [parse_line(line) for line in lines]
        assert {p.measurement for p in points} == {"health"}
        assert [p.timestamp for p in points] == [T0, T0 + NS_PER_S, T0 + 2 * NS_PER_S]
        assert points[0].tags == {"profile": "profile-1", "station": "station-1"}

    def test_a_version_1_write_uses_the_database_and_basic_authentication(
        self, server: FakeInfluxServer
    ) -> None:
        sink = make_sink(server, version=1, database="seeing", username=USER, org=None, bucket=None)
        sink.send("health", health_rows(1))
        (request,) = server.requests
        assert request.path == "/write"
        assert request.query == {"db": ["seeing"], "precision": ["ns"]}
        expected = base64.b64encode(f"{USER}:{CODE}".encode()).decode()
        assert request.headers["authorization"] == f"Basic {expected}"

    def test_a_version_1_write_can_name_a_retention_policy_and_skip_authentication(
        self, server: FakeInfluxServer
    ) -> None:
        sink = make_sink(
            server, version=1, database="seeing", retention_policy="autogen", org=None, bucket=None
        )
        sink.send("health", health_rows(1))
        (request,) = server.requests
        assert request.query["rp"] == ["autogen"]
        assert "authorization" not in request.headers

    def test_a_version_2_write_without_a_token_sends_no_authorization_header(
        self, server: FakeInfluxServer
    ) -> None:
        sink = InfluxSink(
            "influx", config(server), opener=make_opener(use_environment_proxies=False)
        )
        sink.send("health", health_rows(1))
        assert "authorization" not in server.requests[0].headers

    def test_the_same_batch_sends_the_same_body_so_a_repeat_changes_nothing(
        self, server: FakeInfluxServer
    ) -> None:
        sink = make_sink(server)
        rows = health_rows(4)
        sink.send("health", rows)
        sink.send("health", rows)
        assert server.requests[0].body == server.requests[1].body

    def test_a_trailing_slash_in_the_endpoint_does_not_double(
        self, server: FakeInfluxServer
    ) -> None:
        make_sink(server, endpoint=server.url + "/").send("health", health_rows(1))
        assert server.requests[0].path == "/api/v2/write"

    def test_the_sink_reports_its_name_and_batch_size(self, server: FakeInfluxServer) -> None:
        sink = make_sink(server, max_batch_rows=250)
        assert (sink.name, sink.max_batch_rows) == ("influx", 250)
        assert isinstance(sink, Sink)

    def test_it_accepts_every_table_record_type_and_no_segment_type_by_default(
        self, server: FakeInfluxServer
    ) -> None:
        sink = make_sink(server)
        assert sink.accepts("health")
        assert sink.accepts("seeing_window")
        assert not sink.accepts("frame")
        assert not sink.accepts("no_such_type")

    def test_it_accepts_only_the_listed_record_types(self, server: FakeInfluxServer) -> None:
        sink = make_sink(server, record_types=["seeing_window"])
        assert sink.accepts("seeing_window")
        assert not sink.accepts("health")

    def test_the_write_url_holds_no_secret(self, server: FakeInfluxServer) -> None:
        url = InfluxSink.write_url(config(server, token=TOKEN))
        assert TOKEN not in url
        assert url.startswith(server.url)


class TestOutcomes:
    @pytest.mark.parametrize("status", [200, 202, 204])
    def test_a_2xx_reply_is_a_success(self, server: FakeInfluxServer, status: int) -> None:
        server.script(Reply(status))
        make_sink(server).send("health", health_rows(1))

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 429])
    def test_a_server_error_a_timeout_status_and_throttling_are_retryable(
        self, server: FakeInfluxServer, status: int
    ) -> None:
        server.script(Reply(status, '{"message": "try later"}'))
        with pytest.raises(SinkError) as caught:
            make_sink(server).send("health", health_rows(1))
        assert caught.value.retryable is True
        assert f"HTTP {status}" in str(caught.value)
        assert "try later" in str(caught.value)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 413, 422])
    def test_any_other_client_error_is_permanent(
        self, server: FakeInfluxServer, status: int
    ) -> None:
        server.script(Reply(status, '{"code": "invalid", "message": "bad request"}'))
        with pytest.raises(SinkError) as caught:
            make_sink(server).send("health", health_rows(1))
        assert caught.value.retryable is False
        assert f"HTTP {status}" in str(caught.value)

    def test_a_401_and_a_413_say_what_to_do(self, server: FakeInfluxServer) -> None:
        server.script(Reply(401, '{"error": "unauthorized"}'), Reply(413))
        sink = make_sink(server)
        with pytest.raises(SinkError, match="check the token"):
            sink.send("health", health_rows(1))
        with pytest.raises(SinkError, match="lower max_batch_rows"):
            sink.send("health", health_rows(1))

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    def test_a_redirect_is_permanent_and_is_never_followed(
        self, server: FakeInfluxServer, status: int
    ) -> None:
        server.script(Reply(status, headers={"Location": server.url + "/elsewhere"}))
        with pytest.raises(SinkError) as caught:
            make_sink(server).send("health", health_rows(1))
        assert caught.value.retryable is False
        assert "set the endpoint" in str(caught.value)
        assert len(server.requests) == 1  # the write did not turn into a read of another address

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (
                422,
                '{"code":"unprocessable entity","message":"failure writing points to database: '
                'partial write: points beyond retention policy dropped=3"}',
            ),
            (400, '{"error":"partial write: points beyond retention policy dropped=1"}'),
        ],
    )
    def test_points_beyond_the_retention_policy_are_dropped_by_the_server_not_retried(
        self, server: FakeInfluxServer, status: int, body: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        server.script(Reply(status, body))
        with caplog.at_level("WARNING"):
            make_sink(server).send("health", health_rows(1))  # no error: the sink goes on
        assert "beyond the retention policy" in caplog.text

    def test_a_server_that_is_gone_is_a_retryable_network_error(self) -> None:
        with FakeInfluxServer() as gone:
            sink = make_sink(gone)
        with pytest.raises(SinkError) as caught:
            sink.send("health", health_rows(1))  # nothing listens on the port any more
        assert caught.value.retryable is True
        assert "not reachable" in str(caught.value)

    def test_a_connection_closed_without_a_reply_is_retryable(
        self, server: FakeInfluxServer
    ) -> None:
        server.script(Drop())
        with pytest.raises(SinkError) as caught:
            make_sink(server).send("health", health_rows(1))
        assert caught.value.retryable is True

    def test_a_reply_slower_than_the_timeout_is_retryable(self, server: FakeInfluxServer) -> None:
        server.script(Hang(10))
        sink = make_sink(server, timeout_s=0.3)
        started = time.monotonic()
        with pytest.raises(SinkError) as caught:
            sink.send("health", health_rows(1))
        assert time.monotonic() - started < 5
        assert caught.value.retryable is True

    def test_a_slow_reply_inside_the_timeout_succeeds(self, server: FakeInfluxServer) -> None:
        server.script(Hang(0.2))
        make_sink(server, timeout_s=5).send("health", health_rows(1))
        assert len(server.requests) == 1

    def test_an_error_body_that_is_not_json_still_gives_a_short_message(
        self, server: FakeInfluxServer
    ) -> None:
        server.script(Reply(500, "x" * 5000))
        with pytest.raises(SinkError) as caught:
            make_sink(server).send("health", health_rows(1))
        assert len(str(caught.value)) < 400


FAILURES: list[Any] = [
    Reply(401, '{"message": "unauthorized"}'),
    Reply(500),
    Reply(413),
    Reply(302, headers={"Location": "/elsewhere"}),
    Drop(),
]


class TestSecretsStayOut:
    @pytest.mark.parametrize(
        "behavior", FAILURES, ids=lambda b: type(b).__name__ + str(getattr(b, "status", ""))
    )
    def test_no_error_holds_the_token_or_the_address(
        self, server: FakeInfluxServer, behavior: Any
    ) -> None:
        server.script(behavior)
        sink = make_sink(server)
        with pytest.raises(SinkError) as caught:
            sink.send("health", health_rows(1))
        text = str(caught.value) + repr(caught.value)
        assert TOKEN not in text
        assert server.url not in text
        assert TOKEN not in repr(sink)

    def test_a_version_1_error_holds_no_password(self, server: FakeInfluxServer) -> None:
        server.script(Reply(401, '{"error": "authorization failed"}'))
        sink = make_sink(server, version=1, database="seeing", username=USER, org=None, bucket=None)
        with pytest.raises(SinkError) as caught:
            sink.send("health", health_rows(1))
        assert CODE not in str(caught.value)
        assert CODE not in repr(sink)


def handlers_of(opener: urllib.request.OpenerDirector) -> list[Any]:
    return list(vars(opener)["handlers"])


class TestOpener:
    def test_the_opener_can_skip_the_certificate_check(self) -> None:
        opener = make_opener(verify_tls=False)
        contexts = [
            handler._context  # type: ignore[attr-defined]
            for handler in handlers_of(opener)
            if isinstance(handler, urllib.request.HTTPSHandler)
        ]
        assert contexts
        assert contexts[0].verify_mode == ssl.CERT_NONE
        assert contexts[0].check_hostname is False

    def test_the_default_opener_checks_certificates(self) -> None:
        opener = make_opener()
        verified = [
            handler._context.verify_mode  # type: ignore[attr-defined]
            for handler in handlers_of(opener)
            if isinstance(handler, urllib.request.HTTPSHandler)
        ]
        assert all(mode == ssl.CERT_REQUIRED for mode in verified)


class TestWithTheForwarder:
    def test_a_server_that_fails_and_returns_receives_every_row_in_order(
        self, tmp_path: Any, server: FakeInfluxServer
    ) -> None:
        clock = VirtualClock(T0)
        with Store.open(tmp_path / "results.sqlite") as store:
            store.write_many([make_health(T0 + n * NS_PER_S) for n in range(250)])
            store.write_many([make_window(T0 + n * NS_PER_S) for n in range(0, 250, 10)])
            sink = make_sink(server, max_batch_rows=100)
            forwarder = Forwarder(
                store,
                [sink],
                clock,
                ForwarderConfig(jitter=0, backoff_initial_s=1),
                rng=random.Random(1),
            )
            server.script(Reply(503), Reply(503), Reply(500), Reply(429), Drop())
            for _ in range(200):
                forwarder.run_once()
                if not any(forwarder.sink_backlog().values()):
                    break
                clock.advance(2)
            assert forwarder.sink_backlog() == {"influx": 0}
        points = [
            parse_line(line) for request in server.requests for line in split_lines(request.body)
        ]
        by_measurement: dict[str, list[int]] = {}
        for point in points:
            assert point.timestamp is not None
            by_measurement.setdefault(point.measurement, []).append(point.timestamp)
        # Every row arrived. A failed request was sent again in full, and the server upserts by
        # key (measurement, tags, time), so a repeat changes nothing.
        assert set(by_measurement["health"]) == {T0 + n * NS_PER_S for n in range(250)}
        assert set(by_measurement["seeing_window"]) == {
            T0 + n * NS_PER_S for n in range(0, 250, 10)
        }
        for stamps in by_measurement.values():
            first_sightings = list(dict.fromkeys(stamps))  # a repeated batch repeats its rows
            assert first_sightings == sorted(first_sightings)  # the first delivery keeps the order
        assert sum(1 for r in server.requests if r.body) == len(server.requests)
