"""`create_sqm_reader` and `read_sqm_once`: the source picks the reader, and the secrets come in."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.config import ConfigError
from seeingmon.hardware.sqm import SqmConfig, SqmConnectionError, SqmInfluxConfig, SqmLeReader
from seeingmon.hardware.sqm_factory import SqmSample, create_sqm_reader, read_sqm_once
from seeingmon.hardware.sqm_influx import (
    InfluxStaleError,
    InfluxUnauthorizedError,
    SqmInfluxReader,
)
from seeingmon.sinks.influx import make_opener
from tests.hardware.influx_replies import MAGNITUDE, TEMPERATURE, ago, v1_reply, v2_csv
from tests.hardware.sqm_server import FakeSqmServer
from tests.sinks.fake_influx import FakeInfluxServer, Reply

TOKEN = "example-token-value"
CODE = "example-code-value"  # the password of a version 1 server
CODE_VARIABLE = "THE_CODE_VARIABLE"  # the variable that holds it


@pytest.fixture
def server() -> Iterator[FakeInfluxServer]:
    with FakeInfluxServer() as running:
        yield running


def config(server: FakeInfluxServer, *, version: int = 2, **influx: Any) -> SqmConfig:
    values: dict[str, Any] = {
        "endpoint": server.url,
        "timeout_s": 5.0,
        "measurement": "sqm",
        "field": "mag",
        "temperature_field": "temp",
    }
    if version == 2:
        values.update(org="example-org", bucket="example-bucket")
    else:
        values.update(version=1, database="example-db", username="example-user")
    values.update(influx)
    return SqmConfig(source="influx", influx=SqmInfluxConfig(**values))


class TestCreate:
    def test_the_tcp_source_builds_the_tcp_reader(self) -> None:
        reader = create_sqm_reader(
            SqmConfig(enabled=True, host="unit.example.org"),
            clock=VirtualClock(),
            station_id="s",
            profile_id="p",
        )
        assert isinstance(reader, SqmLeReader)

    def test_the_influx_source_builds_the_influx_reader(self, server: FakeInfluxServer) -> None:
        reader = create_sqm_reader(
            config(server), clock=VirtualClock(), station_id="s", profile_id="p", env={}
        )
        assert isinstance(reader, SqmInfluxReader)

    def test_a_token_comes_from_the_variable_that_token_env_names(
        self, server: FakeInfluxServer
    ) -> None:
        clock = VirtualClock()
        reader = create_sqm_reader(
            config(server, token_env="THE_TOKEN_VARIABLE"),
            clock=clock,
            station_id="s",
            profile_id="p",
            env={"THE_TOKEN_VARIABLE": TOKEN},
            opener=make_opener(use_environment_proxies=False),
        )
        server.script(Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 5), MAGNITUDE)])))
        assert reader.poll() is not None
        assert server.requests[0].headers["authorization"] == f"Token {TOKEN}"

    def test_a_token_in_the_configuration_works_too(self, server: FakeInfluxServer) -> None:
        clock = VirtualClock()
        reader = create_sqm_reader(
            config(server, token=TOKEN),
            clock=clock,
            station_id="s",
            profile_id="p",
            env={},
            opener=make_opener(use_environment_proxies=False),
        )
        server.script(Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 5), MAGNITUDE)])))
        reader.poll()
        assert server.requests[0].headers["authorization"] == f"Token {TOKEN}"

    def test_a_password_comes_from_the_variable_that_password_env_names(
        self, server: FakeInfluxServer
    ) -> None:
        clock = VirtualClock()
        reader = create_sqm_reader(
            config(server, version=1, password_env=CODE_VARIABLE),
            clock=clock,
            station_id="s",
            profile_id="p",
            env={CODE_VARIABLE: CODE},
            opener=make_opener(use_environment_proxies=False),
        )
        server.script(Reply(200, v1_reply([[ago(clock.utc_ns(), 5), MAGNITUDE, None]])))
        assert reader.poll() is not None
        expected = base64.b64encode(f"example-user:{CODE}".encode()).decode()
        assert server.requests[0].headers["authorization"] == f"Basic {expected}"

    @pytest.mark.parametrize("env", [{}, {"THE_TOKEN_VARIABLE": ""}])
    def test_a_variable_that_is_not_set_is_a_config_error_that_names_the_variable(
        self, server: FakeInfluxServer, env: dict[str, str]
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            create_sqm_reader(
                config(server, token_env="THE_TOKEN_VARIABLE"),
                clock=VirtualClock(),
                station_id="s",
                profile_id="p",
                env=env,
            )
        assert "THE_TOKEN_VARIABLE" in str(caught.value)
        assert "the token of [sqm.influx]" in str(caught.value)

    def test_the_variables_come_from_the_process_environment_by_default(
        self, server: FakeInfluxServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("THE_TOKEN_VARIABLE", TOKEN)
        reader = create_sqm_reader(
            config(server, token_env="THE_TOKEN_VARIABLE"),
            clock=VirtualClock(),
            station_id="s",
            profile_id="p",
        )
        assert isinstance(reader, SqmInfluxReader)

    def test_the_source_influx_without_the_table_is_a_config_error(self) -> None:
        with pytest.raises(ConfigError, match=r"needs the table \[sqm.influx\]"):
            create_sqm_reader(
                SqmConfig(source="influx"), clock=VirtualClock(), station_id="s", profile_id="p"
            )


class TestReadOnce:
    def test_an_influx_reading_has_the_magnitude_the_temperature_and_the_age(
        self, server: FakeInfluxServer
    ) -> None:
        clock = VirtualClock()
        server.script(
            Reply(
                200,
                v2_csv(
                    [
                        ("mag", ago(clock.utc_ns(), 12.0), MAGNITUDE),
                        ("temp", ago(clock.utc_ns(), 12.0), TEMPERATURE),
                    ]
                ),
            )
        )
        sample = read_sqm_once(
            config(server, token=TOKEN),
            clock=clock,
            env={},
            opener=make_opener(use_environment_proxies=False),
        )
        assert sample == SqmSample("influx", MAGNITUDE, TEMPERATURE, 12.0)
        assert sample.summary() == (
            "magnitude 21.43 mag/arcsec^2, temperature 3.4 C, age 12.0 s (source influx)"
        )

    def test_the_reading_does_not_need_the_reader_to_be_enabled(
        self, server: FakeInfluxServer
    ) -> None:
        clock = VirtualClock()
        settings = config(server, token=TOKEN)
        assert settings.enabled is False
        server.script(Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 3.0), MAGNITUDE)])))
        sample = read_sqm_once(
            settings, clock=clock, env={}, opener=make_opener(use_environment_proxies=False)
        )
        assert sample.temperature_c is None
        assert sample.summary().startswith("magnitude 21.43 mag/arcsec^2, temperature n/a, age 3.0")

    def test_a_failed_read_raises_the_typed_error_of_the_reader(
        self, server: FakeInfluxServer
    ) -> None:
        clock = VirtualClock()
        settings = config(server, token=TOKEN)
        opener = make_opener(use_environment_proxies=False)
        server.script(Reply(401, '{"message": "unauthorized access"}'))
        with pytest.raises(InfluxUnauthorizedError):
            read_sqm_once(settings, clock=clock, env={}, opener=opener)
        server.script(Reply(200, v2_csv([("mag", ago(clock.utc_ns(), 4000.0), MAGNITUDE)])))
        with pytest.raises(InfluxStaleError):
            read_sqm_once(settings, clock=clock, env={}, opener=opener)

    def test_a_tcp_reading_has_age_zero(self) -> None:
        fake = FakeSqmServer()
        try:
            settings = SqmConfig(host="127.0.0.1", port=fake.port, read_timeout_s=1.0)
            sample = read_sqm_once(settings, clock=VirtualClock())
        finally:
            fake.close()
        assert sample == SqmSample("tcp", 21.37, 3.5, 0.0)
        assert sample.summary() == (
            "magnitude 21.37 mag/arcsec^2, temperature 3.5 C, age 0.0 s (source tcp)"
        )
        assert fake.commands == [b"rx"]

    def test_a_tcp_source_without_a_host_is_a_config_error(self) -> None:
        with pytest.raises(ConfigError, match=r"host is not set.*\[sqm.influx\]"):
            read_sqm_once(SqmConfig(), clock=VirtualClock())

    def test_a_tcp_failure_raises_the_tcp_error(self) -> None:
        fake = FakeSqmServer()
        fake.script.append("drop")
        try:
            settings = SqmConfig(host="127.0.0.1", port=fake.port, read_timeout_s=1.0)
            with pytest.raises(SqmConnectionError):
                read_sqm_once(settings, clock=VirtualClock())
        finally:
            fake.close()

    def test_the_summary_reads_the_same_for_a_reading_without_a_temperature(self) -> None:
        sample = SqmSample("influx", 18.5, None, 61.04)
        assert sample.summary() == (
            "magnitude 18.50 mag/arcsec^2, temperature n/a, age 61.0 s (source influx)"
        )
