"""The `[sinks.<name>]` tables: validation, secrets, and the environment scheme."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from seeingmon.config import Config, ConfigError, load_config
from seeingmon.sinks.config import (
    InfluxSinkConfig,
    SinksSection,
    TimescaleSinkConfig,
    check_endpoint,
    check_influx_connection,
    resolve_credential,
)

ENDPOINT = "https://influx.example.org:8086"
TOKEN = "example-token-value"
CODE = "example-code-value"


def influx(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "kind": "influx",
        "endpoint": ENDPOINT,
        "org": "example-org",
        "bucket": "seeing",
    }
    values.update(overrides)
    return values


def timescale(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "kind": "timescale",
        "host": "timescale.example.org",
        "database": "seeing",
        "user": "example-user",
    }
    values.update(overrides)
    return values


class TestInflux:
    def test_a_version_2_sink_needs_an_org_and_a_bucket_and_has_sensible_defaults(self) -> None:
        sink = InfluxSinkConfig(**influx())
        assert sink.version == 2
        assert sink.enabled is True
        assert sink.record_types is None
        assert sink.timeout_s == 10
        assert sink.verify_tls is True
        assert sink.max_batch_rows == 5000
        assert sink.token is None

    def test_the_endpoint_loses_a_trailing_slash(self) -> None:
        assert InfluxSinkConfig(**influx(endpoint=ENDPOINT + "/")).endpoint == ENDPOINT

    @pytest.mark.parametrize(
        "endpoint",
        [
            "private-marker.example.net",
            "ftp://private-marker.example.net",
            "https://",
            "https://user.example.net:"
            + "private-marker@host.example.net",  # credentials in the address
            "https://host.example.net/?org=private-marker",
            "https://host.example.net/#private-marker",
        ],
    )
    def test_a_bad_endpoint_is_rejected_without_echoing_it(self, endpoint: str) -> None:
        with pytest.raises(ValidationError) as caught:
            InfluxSinkConfig(**influx(endpoint=endpoint))
        for error in caught.value.errors(include_input=False):
            assert "private-marker" not in error["msg"]

    def test_version_1_needs_a_database_and_takes_a_user_and_a_password(self) -> None:
        sink = InfluxSinkConfig(
            kind="influx",
            version=1,
            endpoint=ENDPOINT,
            database="seeing",
            username="example-user",
            password=CODE,
        )
        assert sink.password is not None
        assert sink.password.get_secret_value() == CODE
        with pytest.raises(ValidationError, match="needs a database"):
            InfluxSinkConfig(kind="influx", version=1, endpoint=ENDPOINT)

    def test_version_1_has_no_token_and_version_2_has_no_password(self) -> None:
        with pytest.raises(ValidationError, match="no token"):
            InfluxSinkConfig(kind="influx", version=1, endpoint=ENDPOINT, database="d", token=TOKEN)
        with pytest.raises(ValidationError, match="not a password"):
            InfluxSinkConfig(**influx(username="u", password="p"))

    def test_version_2_needs_an_org_and_a_bucket(self) -> None:
        with pytest.raises(ValidationError, match="org and a bucket"):
            InfluxSinkConfig(kind="influx", endpoint=ENDPOINT, bucket="seeing")

    def test_a_secret_has_one_source(self) -> None:
        with pytest.raises(ValidationError, match="token or token_env"):
            InfluxSinkConfig(**influx(token=TOKEN, token_env="SOME_VARIABLE"))
        with pytest.raises(ValidationError, match="password or password_env"):
            InfluxSinkConfig(
                kind="influx",
                version=1,
                endpoint=ENDPOINT,
                database="d",
                username="u",
                password="p",
                password_env="X",
            )
        with pytest.raises(ValidationError, match="needs a username"):
            InfluxSinkConfig(
                kind="influx", version=1, endpoint=ENDPOINT, database="d", password_env="X"
            )

    def test_a_secret_stays_out_of_the_repr(self) -> None:
        sink = InfluxSinkConfig(**influx(token=TOKEN))
        assert TOKEN not in repr(sink)
        assert TOKEN not in str(sink)
        assert TOKEN not in sink.model_dump_json()

    @pytest.mark.parametrize("key", ["timeout_s", "max_batch_rows"])
    def test_numbers_have_bounds(self, key: str) -> None:
        with pytest.raises(ValidationError):
            InfluxSinkConfig(**influx(**{key: 0}))

    def test_an_unknown_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="tokn"):
            InfluxSinkConfig(**influx(tokn="x"))


class TestTimescale:
    def test_the_defaults(self) -> None:
        sink = TimescaleSinkConfig(**timescale())
        assert (sink.port, sink.sslmode, sink.hypertables) == (5432, "prefer", None)
        assert (sink.chunk_days, sink.max_batch_rows) == (7, 1000)

    def test_the_ssl_mode_is_one_of_the_postgres_modes(self) -> None:
        assert TimescaleSinkConfig(**timescale(sslmode="verify-full")).sslmode == "verify-full"
        with pytest.raises(ValidationError):
            TimescaleSinkConfig(**timescale(sslmode="sometimes"))

    def test_a_password_has_one_source(self) -> None:
        with pytest.raises(ValidationError, match="password or password_env"):
            TimescaleSinkConfig(**timescale(password="p", password_env="X"))

    def test_the_port_is_a_port(self) -> None:
        with pytest.raises(ValidationError):
            TimescaleSinkConfig(**timescale(port=70000))


class TestSection:
    def test_a_missing_section_means_no_sinks(self) -> None:
        assert Config({}).section("sinks", SinksSection).root == {}

    def test_each_table_picks_its_kind(self) -> None:
        config = Config({"sinks": {"lab_influx": influx(), "archive": timescale()}})
        sinks = config.section("sinks", SinksSection).root
        assert isinstance(sinks["lab_influx"], InfluxSinkConfig)
        assert isinstance(sinks["archive"], TimescaleSinkConfig)

    def test_an_unknown_kind_is_an_error(self) -> None:
        config = Config({"sinks": {"x": {"kind": "carrier-pigeon"}}})
        with pytest.raises(ConfigError, match="carrier-pigeon"):
            config.section("sinks", SinksSection)

    @pytest.mark.parametrize("name", ["Lab", "1lab", "lab influx", "x" * 33, "lab.influx"])
    def test_a_sink_name_is_a_short_lowercase_word(self, name: str) -> None:
        with pytest.raises(ConfigError, match="sink name"):
            Config({"sinks": {name: influx()}}).section("sinks", SinksSection)

    def test_an_error_names_the_sink_and_the_key_and_never_a_secret(self) -> None:
        config = Config({"sinks": {"lab": influx(token=TOKEN, tokn="oops", endpoint="nope")}})
        with pytest.raises(ConfigError) as caught:
            config.section("sinks", SinksSection)
        message = str(caught.value)
        assert "lab" in message
        assert TOKEN not in message
        assert "oops" not in message

    def test_the_record_types_are_checked_against_the_declarations(self) -> None:
        sink = InfluxSinkConfig(**influx(record_types=["health", "seeing_window"]))
        assert sink.record_types == ["health", "seeing_window"]
        with pytest.raises(ValidationError, match="not a record type"):
            InfluxSinkConfig(**influx(record_types=["nonsense"]))
        with pytest.raises(ValidationError, match="segment files"):
            InfluxSinkConfig(**influx(record_types=["frame"]))

    def test_a_sink_can_be_switched_off(self) -> None:
        assert InfluxSinkConfig(**influx(enabled=False)).enabled is False


class TestEnvironmentScheme:
    def test_environment_variables_build_a_whole_sink_and_a_number_stays_text(
        self, tmp_path: Path
    ) -> None:
        env = {
            "SEEINGMON_SINKS__LAB_INFLUX__KIND": "influx",
            "SEEINGMON_SINKS__LAB_INFLUX__ENDPOINT": ENDPOINT,
            "SEEINGMON_SINKS__LAB_INFLUX__ORG": "example-org",
            "SEEINGMON_SINKS__LAB_INFLUX__BUCKET": "seeing",
            "SEEINGMON_SINKS__LAB_INFLUX__TOKEN": "12345",  # parses as a TOML integer
            "SEEINGMON_SINKS__LAB_INFLUX__TIMEOUT_S": "3",
        }
        config = load_config(local_file=tmp_path / "none.toml", env=env)
        (sink,) = config.section("sinks", SinksSection).root.values()
        assert isinstance(sink, InfluxSinkConfig)
        assert sink.timeout_s == 3
        assert sink.token is not None
        assert sink.token.get_secret_value() == "12345"

    def test_the_effective_configuration_redacts_the_secrets(self, tmp_path: Path) -> None:
        local = tmp_path / "config.toml"
        local.write_text(
            f'[sinks.lab]\nkind = "influx"\nendpoint = "{ENDPOINT}"\norg = "o"\nbucket = "b"\n'
            f'token = "{TOKEN}"\n',
            encoding="utf-8",
        )
        config = load_config(local_file=local, env={})
        effective = config.effective()
        assert effective["sinks"]["lab"]["token"] == "<redacted>"
        assert TOKEN not in str(effective)


class TestResolveSecret:
    def test_an_environment_variable_supplies_the_secret(self) -> None:
        assert resolve_credential(None, "THE_VARIABLE", {"THE_VARIABLE": "value-1"}, "a token") == (
            "value-1"
        )

    def test_a_direct_value_supplies_the_secret(self) -> None:
        config = InfluxSinkConfig(**influx(token=TOKEN))
        assert resolve_credential(config.token, None, {}, "a token") == TOKEN

    def test_no_secret_gives_none(self) -> None:
        assert resolve_credential(None, None, {}, "a token") is None

    @pytest.mark.parametrize("env", [{}, {"THE_VARIABLE": ""}])
    def test_a_missing_variable_is_an_error_that_names_the_variable(
        self, env: dict[str, str]
    ) -> None:
        with pytest.raises(ConfigError, match="THE_VARIABLE"):
            resolve_credential(None, "THE_VARIABLE", env, "the token of sink lab")


class TestSharedChecks:
    """The checks of an InfluxDB connection that other readers of the system use too."""

    def test_the_endpoint_check_returns_the_address_without_a_trailing_slash(self) -> None:
        assert check_endpoint(ENDPOINT + "/") == ENDPOINT
        assert check_endpoint("http://localhost:8086") == "http://localhost:8086"

    @pytest.mark.parametrize(
        "endpoint",
        ["private-marker.example.net", "ftp://private-marker.example.net", "https://"],
    )
    def test_a_bad_endpoint_raises_without_echoing_it(self, endpoint: str) -> None:
        with pytest.raises(ValueError, match="http") as caught:
            check_endpoint(endpoint)
        assert "private-marker" not in str(caught.value)

    def test_the_connection_check_asks_for_the_keys_of_the_version(self) -> None:
        keys: dict[str, Any] = {
            "org": None,
            "bucket": None,
            "database": None,
            "token": None,
            "token_env": None,
            "username": None,
            "password": None,
            "password_env": None,
        }
        check_influx_connection(version=2, **{**keys, "org": "o", "bucket": "b"})
        check_influx_connection(version=1, **{**keys, "database": "d"})
        with pytest.raises(ValueError, match="org and a bucket"):
            check_influx_connection(version=2, **keys)
        with pytest.raises(ValueError, match="needs a database"):
            check_influx_connection(version=1, **keys)
        with pytest.raises(ValueError, match="needs a username"):
            check_influx_connection(version=1, **{**keys, "database": "d", "password_env": "X"})
