"""The SQM-LE reader in `core`: `[sqm] source` picks the reader, and its readings reach the store.

The tests build `CoreApp` on fakes and a `VirtualClock` (see `rig`), and a fake InfluxDB answers the
reader. A stepped run polls the reader through the periodic tasks, so a test moves virtual time
and calls `tick`. One test runs the thread of the reader against a clock that runs faster than real
time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")

from seeingmon.clock import NS_PER_S, ScaledClock
from seeingmon.config import ConfigError
from seeingmon.hardware import sqm_influx
from seeingmon.hardware.sqm import SqmLeReader
from seeingmon.hardware.sqm_influx import SqmInfluxReader
from seeingmon.records import HealthRecord, ReferenceRecord
from seeingmon.sinks.influx import make_opener
from tests.hardware.influx_replies import MAGNITUDE, TEMPERATURE, v2_csv
from tests.sinks.fake_influx import FakeInfluxServer, Reply, Respond

from .rig import NIGHT, CoreRig, build_rig

TOKEN = "example-token-value"


@pytest.fixture
def server() -> Iterator[FakeInfluxServer]:
    with FakeInfluxServer() as running:
        yield running


@pytest.fixture(autouse=True)
def _loopback_without_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core builds the HTTP client of the reader itself, so keep the machine's proxies out."""
    monkeypatch.setattr(
        sqm_influx, "make_opener", partial(make_opener, use_environment_proxies=False)
    )


def influx_text(server: FakeInfluxServer, *, secret: str = f'token = "{TOKEN}"\n') -> str:
    return (
        '[sqm]\nenabled = true\nsource = "influx"\naltitude_deg = 45.0\nazimuth_deg = 0.0\n'
        f'[sqm.influx]\nendpoint = "{server.url}"\norg = "example-org"\n'
        f'bucket = "example-bucket"\n{secret}'
        'measurement = "sqm"\nfield = "mag"\ntemperature_field = "temp"\n'
        '[sqm.influx.tags]\nunit = "roof"\n'
    )


def point_reply(rig: CoreRig, age_s: float = 5.0) -> Reply:
    """A point that is `age_s` old on the clock of the rig, as InfluxDB 2 sends it."""
    t_utc_ns = rig.clock.utc_ns() - round(age_s * NS_PER_S)
    return Reply(200, v2_csv([("mag", t_utc_ns, MAGNITUDE), ("temp", t_utc_ns, TEMPERATURE)]))


def health_components(rig: CoreRig) -> dict[str, str]:
    """Ask for a health record at once, and return its components.

    A record has the key of its time, so the clock moves a millisecond first: the tick that came
    before may have written a health record at this very instant.
    """
    rig.clock.advance(0.001)
    rig.app.tasks.trigger("health")
    rig.app.tick()
    record = rig.records("health")[-1]
    assert isinstance(record, HealthRecord)
    return dict(record.components)


class TestWiring:
    def test_the_source_influx_builds_the_reader_of_influxdb(
        self, tmp_path: Path, server: FakeInfluxServer
    ) -> None:
        rig = build_rig(tmp_path, config_extra=influx_text(server))
        try:
            assert isinstance(rig.app.sqm, SqmInfluxReader)
        finally:
            rig.app.stop()

    def test_the_source_tcp_builds_the_tcp_reader(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path, config_extra='[sqm]\nenabled = true\nhost = "unit.example.org"\n')
        try:
            assert isinstance(rig.app.sqm, SqmLeReader)
        finally:
            rig.app.stop()

    def test_a_reader_that_is_off_is_not_built(self, tmp_path: Path) -> None:
        rig = build_rig(tmp_path)
        try:
            assert rig.app.sqm is None
            rig.app.start()
            rig.app.tick()
            assert "sqm" not in health_components(rig)
        finally:
            rig.app.stop()

    def test_a_token_variable_that_is_not_set_stops_the_start_and_names_the_variable(
        self, tmp_path: Path, server: FakeInfluxServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CORE_TEST_TOKEN_VARIABLE", raising=False)
        text = influx_text(server, secret='token_env = "CORE_TEST_TOKEN_VARIABLE"\n')
        with pytest.raises(ConfigError, match="CORE_TEST_TOKEN_VARIABLE"):
            build_rig(tmp_path, config_extra=text)

    def test_a_token_variable_that_is_set_authorizes_the_queries(
        self, tmp_path: Path, server: FakeInfluxServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORE_TEST_TOKEN_VARIABLE", TOKEN)
        text = influx_text(server, secret='token_env = "CORE_TEST_TOKEN_VARIABLE"\n')
        rig = build_rig(tmp_path, config_extra=text)
        try:
            server.default = Respond(lambda request: point_reply(rig))
            rig.app.start()
            rig.app.tick()
            assert server.requests[0].headers["authorization"] == f"Token {TOKEN}"
        finally:
            rig.app.stop()

    def test_the_run_record_hides_the_secrets_the_endpoint_and_the_names_of_the_data(
        self, tmp_path: Path, server: FakeInfluxServer
    ) -> None:
        rig = build_rig(tmp_path, config_extra=influx_text(server))
        try:
            server.default = Respond(lambda request: point_reply(rig))
            rig.app.start()
            rig.app.scheduler.step()  # the first step opens the camera, and the run record follows
            (run,) = rig.records("run")
            config = run.effective_config  # type: ignore[attr-defined]
            text = str(config)
            for value in (TOKEN, server.url, "example-org", "example-bucket", "roof"):
                assert value not in text
            assert config["sqm"]["influx"]["measurement"] == "<redacted>"
            assert config["sqm"]["influx"]["tags"] == "<redacted>"
            assert config["sqm"]["source"] == "influx"
        finally:
            rig.app.stop()


class TestSteppedRun:
    def test_each_new_point_becomes_a_reference_record_in_the_store(
        self, tmp_path: Path, server: FakeInfluxServer
    ) -> None:
        rig = build_rig(tmp_path, config_extra=influx_text(server))
        try:
            server.default = Respond(lambda request: point_reply(rig, 5.0))
            rig.app.start()
            rig.app.tick()
            (first,) = rig.records("reference")
            assert isinstance(first, ReferenceRecord)
            assert first.t_utc_ns == NIGHT - 5 * NS_PER_S  # the time of the point
            assert (first.instrument, first.source) == ("sqm_le", "fixed")
            assert (first.value_mag_arcsec2, first.temperature_c) == (MAGNITUDE, TEMPERATURE)
            assert (first.altitude_deg, first.azimuth_deg) == (45.0, 0.0)
            assert first.station_id == "test"
            assert first.provenance == {"reader": "sqm-influx-1", "api": "2"}
            rig.clock.advance(30.0)
            rig.app.tick()
            assert len(rig.records("reference")) == 1  # the poll interval is 60 s
            rig.clock.advance(30.0)
            rig.app.tick()
            assert len(rig.records("reference")) == 2
            assert len(server.requests) == 2
        finally:
            rig.app.stop()

    def test_a_point_that_repeats_adds_no_record_and_no_failure(
        self, tmp_path: Path, server: FakeInfluxServer
    ) -> None:
        rig = build_rig(tmp_path, config_extra=influx_text(server))
        try:
            fixed = point_reply(rig, 5.0)
            server.default = fixed
            rig.app.start()
            for _ in range(4):
                rig.app.tick()
                rig.clock.advance(60.0)
            assert len(rig.records("reference")) == 1
            assert len(server.requests) == 4
            assert rig.events("sqm.read_failed") == []
            assert health_components(rig)["sqm"] == "ok"
        finally:
            rig.app.stop()

    def test_health_reports_the_sqm_component_with_the_counts_of_the_tcp_reader(
        self, tmp_path: Path, server: FakeInfluxServer
    ) -> None:
        rig = build_rig(tmp_path, config_extra=influx_text(server))
        try:
            server.default = Reply(503)
            rig.app.start()
            states = []
            for wait_s in (0.0, 5.0, 10.0, 20.0, 40.0):  # the backoff after each failure
                rig.clock.advance(wait_s)
                rig.app.tick()
                states.append(health_components(rig)["sqm"])
            assert states == ["degraded", "degraded", "degraded", "degraded", "failed"]
            server.default = Respond(lambda request: point_reply(rig))
            rig.clock.advance(80.0)
            rig.app.tick()
            assert health_components(rig)["sqm"] == "ok"
            (failed,) = rig.events("sqm.read_failed")
            assert failed.detail == {"cause": "Unreachable"}
            (recovered,) = rig.events("sqm.recovered")
            assert recovered.detail == {"failures": 5}
            assert len(rig.records("reference")) == 1
        finally:
            rig.app.stop()

    def test_a_stale_point_degrades_the_component_and_names_the_cause(
        self, tmp_path: Path, server: FakeInfluxServer
    ) -> None:
        rig = build_rig(tmp_path, config_extra=influx_text(server))
        try:
            server.default = Respond(lambda request: point_reply(rig, 5000.0))
            rig.app.start()
            rig.app.tick()
            assert health_components(rig)["sqm"] == "degraded"
            (failed,) = rig.events("sqm.read_failed")
            assert failed.detail == {"cause": "Stale"}
            assert rig.records("reference") == []
        finally:
            rig.app.stop()

    def test_after_a_restart_the_point_that_the_store_holds_is_no_error(
        self, tmp_path: Path, server: FakeInfluxServer, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = build_rig(tmp_path, config_extra=influx_text(server))
        try:
            server.default = point_reply(first, 5.0)  # one point, whenever it is asked for
            first.app.start()
            first.app.tick()
            assert len(first.records("reference")) == 1
        finally:
            first.app.stop()
        second = build_rig(tmp_path, config_extra=influx_text(server))  # the same data directory
        try:
            second.app.start()
            with caplog.at_level(logging.DEBUG):
                second.app.tick()
            assert len(second.records("reference")) == 1  # the new reader found the same point
            assert "periodic task sqm failed" not in caplog.text
            assert second.app.tasks.failures("sqm") == 0
            assert second.events("sqm.read_failed") == []
        finally:
            second.app.stop()


class TestThread:
    def test_the_thread_of_the_reader_stores_the_readings_and_stops_with_core(
        self, tmp_path: Path, server: FakeInfluxServer
    ) -> None:
        clock = ScaledClock(start_utc_ns=NIGHT, origin_real_ns=time.time_ns(), speed=20.0)
        text = influx_text(server).replace(
            'source = "influx"', 'source = "influx"\npoll_interval_s = 1.0'
        )
        rig = build_rig(tmp_path, threads=True, clock=clock, config_extra=text)

        def newest(request: object) -> Reply:
            t_utc_ns = clock.utc_ns() - 2 * NS_PER_S
            return Reply(
                200, v2_csv([("mag", t_utc_ns, MAGNITUDE), ("temp", t_utc_ns, TEMPERATURE)])
            )

        server.default = Respond(newest)
        outcome: list[int] = []
        thread = threading.Thread(target=lambda: outcome.append(rig.app.run()))
        thread.start()
        try:
            stop = threading.Event()
            for _ in range(200):
                if len(rig.records("reference")) >= 2:
                    break
                stop.wait(0.05)
            records = rig.records("reference")
            assert len(records) >= 2
            assert all(isinstance(r, ReferenceRecord) for r in records)
            assert len({r.t_utc_ns for r in records}) == len(records)  # one record for each point
        finally:
            rig.app.request_stop("a test")
            thread.join(30.0)
        assert outcome == [0]
