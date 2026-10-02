"""The `source` of `[sqm]` and the table `[sqm.influx]`: validation, secrets, and environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from seeingmon.config import Config, ConfigError, load_config
from seeingmon.hardware.sqm import SqmConfig, SqmInfluxConfig

ENDPOINT = "https://influx.example.org:8086"
TOKEN = "example-token-value"
CODE = "example-code-value"  # the password of a version 1 server
VARIABLE = "SOME_VARIABLE"  # the name of an environment variable


def influx(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "endpoint": ENDPOINT,
        "org": "example-org",
        "bucket": "example-bucket",
        "measurement": "sqm",
        "field": "mag",
    }
    values.update(overrides)
    return values


def version_1(**overrides: Any) -> dict[str, Any]:
    values = influx(version=1, database="example-db", org=None, bucket=None)
    values.update(overrides)
    return values


class TestInfluxTable:
    def test_a_version_2_table_has_sensible_defaults(self) -> None:
        table = SqmInfluxConfig(**influx())
        assert table.version == 2
        assert (table.timeout_s, table.verify_tls) == (10.0, True)
        assert (table.max_age_s, table.lookback_s) == (600.0, 3600.0)
        assert table.temperature_field is None
        assert table.tags == {}
        assert table.token is None

    def test_the_data_keys_have_no_default(self) -> None:
        for key in ("endpoint", "measurement", "field"):
            values = influx()
            del values[key]
            with pytest.raises(ValidationError, match=key):
                SqmInfluxConfig(**values)

    def test_the_endpoint_loses_a_trailing_slash(self) -> None:
        assert SqmInfluxConfig(**influx(endpoint=ENDPOINT + "/")).endpoint == ENDPOINT

    @pytest.mark.parametrize(
        "endpoint",
        [
            "private-marker.example.net",
            "ftp://private-marker.example.net",
            "https://",
            "https://user.example.net:" + "private-marker@host.example.net",
            "https://host.example.net/?org=private-marker",
            "https://host.example.net/#private-marker",
        ],
    )
    def test_a_bad_endpoint_is_rejected_without_echoing_it(self, endpoint: str) -> None:
        with pytest.raises(ValidationError) as caught:
            SqmInfluxConfig(**influx(endpoint=endpoint))
        for error in caught.value.errors(include_input=False):
            assert "private-marker" not in error["msg"]

    def test_version_1_needs_a_database_and_takes_a_user_and_a_password(self) -> None:
        table = SqmInfluxConfig(**version_1(username="example-user", password=CODE))
        assert table.password is not None
        assert table.password.get_secret_value() == CODE
        with pytest.raises(ValidationError, match="needs a database"):
            SqmInfluxConfig(**version_1(database=None))

    def test_version_1_has_no_token_and_version_2_has_no_password(self) -> None:
        with pytest.raises(ValidationError, match="no token"):
            SqmInfluxConfig(**version_1(token=TOKEN))
        with pytest.raises(ValidationError, match="not a password"):
            SqmInfluxConfig(**influx(username="example-user", password=CODE))

    def test_version_2_needs_an_org_and_a_bucket(self) -> None:
        with pytest.raises(ValidationError, match="org and a bucket"):
            SqmInfluxConfig(**influx(bucket=None))

    def test_a_secret_has_one_source(self) -> None:
        with pytest.raises(ValidationError, match="token or token_env"):
            SqmInfluxConfig(**influx(token=TOKEN, token_env=VARIABLE))
        with pytest.raises(ValidationError, match="password or password_env"):
            SqmInfluxConfig(
                **version_1(username="example-user", password=CODE, password_env=VARIABLE)
            )
        with pytest.raises(ValidationError, match="needs a username"):
            SqmInfluxConfig(**version_1(password_env=VARIABLE))

    def test_a_secret_stays_out_of_the_repr_and_the_dump(self) -> None:
        for table in (
            SqmInfluxConfig(**influx(token=TOKEN)),
            SqmInfluxConfig(**version_1(username="example-user", password=CODE)),
        ):
            for text in (repr(table), str(table), table.model_dump_json()):
                assert TOKEN not in text
                assert CODE not in text

    def test_the_shape_of_the_data_is_checked(self) -> None:
        with pytest.raises(ValidationError, match="differ from field"):
            SqmInfluxConfig(**influx(temperature_field="mag"))
        with pytest.raises(ValidationError, match="not be less than max_age_s"):
            SqmInfluxConfig(**influx(max_age_s=600.0, lookback_s=599.0))
        assert SqmInfluxConfig(**influx(max_age_s=600.0, lookback_s=600.0)).lookback_s == 600.0

    @pytest.mark.parametrize(
        "overrides",
        [
            {"timeout_s": 0},
            {"timeout_s": 301},
            {"max_age_s": 0},
            {"lookback_s": 0},
            {"max_age_s": 31 * 86_400.0, "lookback_s": 32 * 86_400.0},
            {"measurement": ""},
            {"field": ""},
            {"temperature_field": ""},
            {"org": ""},
            {"measurement": "x" * 257},
        ],
    )
    def test_numbers_and_names_have_bounds(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            SqmInfluxConfig(**influx(**overrides))

    def test_tags_map_a_name_to_a_value(self) -> None:
        table = SqmInfluxConfig(**influx(tags={"unit": "roof", "site": "a"}))
        assert table.tags == {"unit": "roof", "site": "a"}

    @pytest.mark.parametrize(
        "tags",
        [{"": "roof"}, {"unit": ""}, {f"tag{n}": "x" for n in range(17)}, {"unit": "x" * 257}],
    )
    def test_a_bad_tag_is_rejected(self, tags: dict[str, str]) -> None:
        with pytest.raises(ValidationError):
            SqmInfluxConfig(**influx(tags=tags))

    def test_any_character_is_allowed_in_a_name_because_the_query_escapes_it(self) -> None:
        table = SqmInfluxConfig(**influx(measurement='a"b\\c\nd', tags={"u'n": "${x}"}))
        assert table.measurement == 'a"b\\c\nd'

    def test_an_unknown_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="tokn"):
            SqmInfluxConfig(**influx(tokn="x"))


class TestSource:
    def test_the_default_source_is_tcp_and_there_is_no_influx_table(self) -> None:
        config = SqmConfig()
        assert config.source == "tcp"
        assert config.influx is None

    def test_an_enabled_tcp_reader_needs_a_host_and_no_influx_table(self) -> None:
        with pytest.raises(ValidationError, match="needs a host"):
            SqmConfig(enabled=True)
        with pytest.raises(ValidationError, match="needs a host"):
            SqmConfig(enabled=True, source="tcp", influx=influx())
        assert SqmConfig(enabled=True, host="unit.example.org").source == "tcp"

    def test_an_enabled_influx_reader_needs_the_table_and_no_host(self) -> None:
        with pytest.raises(ValidationError, match=r"needs \[sqm.influx\]"):
            SqmConfig(enabled=True, source="influx")
        config = SqmConfig(enabled=True, source="influx", influx=influx())
        assert config.host == ""
        assert config.influx is not None
        assert config.influx.measurement == "sqm"

    def test_a_reader_that_stays_off_needs_neither(self) -> None:
        assert SqmConfig(source="influx").enabled is False
        assert SqmConfig(source="influx", influx=influx()).enabled is False

    def test_a_tcp_reader_may_keep_an_influx_table_that_it_ignores(self) -> None:
        config = SqmConfig(enabled=True, host="unit.example.org", influx=influx())
        assert config.source == "tcp"
        assert config.influx is not None

    def test_the_source_is_tcp_or_influx(self) -> None:
        with pytest.raises(ValidationError):
            SqmConfig(source="serial")

    def test_the_table_is_validated_when_it_is_there(self) -> None:
        with pytest.raises(ValidationError, match="org and a bucket"):
            SqmConfig(source="influx", influx=influx(org=None))

    def test_the_backoff_rule_of_the_tcp_reader_still_holds(self) -> None:
        with pytest.raises(ValidationError, match="backoff_max_s"):
            SqmConfig(backoff_initial_s=10.0, backoff_max_s=5.0)


class TestFromConfigFiles:
    @staticmethod
    def load(tmp_path: Path, text: str, env: dict[str, str] | None = None) -> Config:
        local = tmp_path / "config.toml"
        local.write_text(text, encoding="utf-8")
        return load_config(local_file=local, env=env or {})

    def test_a_local_file_names_the_influx_source_and_its_tags(self, tmp_path: Path) -> None:
        config = self.load(
            tmp_path,
            f'[sqm]\nenabled = true\nsource = "influx"\naltitude_deg = 45.0\n'
            f'[sqm.influx]\nendpoint = "{ENDPOINT}"\norg = "o"\nbucket = "b"\n'
            f'token_env = "SOME_VARIABLE"\nmeasurement = "sqm"\nfield = "mag"\n'
            f'temperature_field = "temp"\n[sqm.influx.tags]\nunit = "roof"\n',
        )
        sqm = config.section("sqm", SqmConfig)
        assert sqm.source == "influx"
        assert sqm.influx is not None
        assert sqm.influx.tags == {"unit": "roof"}
        assert sqm.influx.temperature_field == "temp"

    def test_a_missing_key_names_the_variable_that_would_set_it(self, tmp_path: Path) -> None:
        config = self.load(
            tmp_path, '[sqm]\nsource = "influx"\n[sqm.influx]\nmeasurement = "sqm"\nfield = "mag"\n'
        )
        with pytest.raises(ConfigError, match="SEEINGMON_SQM__INFLUX__ENDPOINT"):
            config.section("sqm", SqmConfig)

    def test_an_error_names_the_key_and_never_a_value(self, tmp_path: Path) -> None:
        config = self.load(
            tmp_path,
            f'[sqm]\nsource = "influx"\n[sqm.influx]\nendpoint = "{ENDPOINT}"\n'
            f'org = "o"\nbucket = "private-marker"\nmeasurement = "sqm"\nfield = "mag"\n'
            f'token = "{TOKEN}"\ntokn = "oops"\n',
        )
        with pytest.raises(ConfigError) as caught:
            config.section("sqm", SqmConfig)
        message = str(caught.value)
        assert "tokn" in message
        for value in (TOKEN, "oops", "private-marker"):
            assert value not in message

    def test_environment_variables_build_the_whole_table_and_a_number_stays_text(
        self, tmp_path: Path
    ) -> None:
        env = {
            "SEEINGMON_SQM__ENABLED": "true",
            "SEEINGMON_SQM__SOURCE": "influx",
            "SEEINGMON_SQM__INFLUX__ENDPOINT": ENDPOINT,
            "SEEINGMON_SQM__INFLUX__ORG": "example-org",
            "SEEINGMON_SQM__INFLUX__BUCKET": "2026",  # parses as a TOML integer
            "SEEINGMON_SQM__INFLUX__TOKEN": "12345",
            "SEEINGMON_SQM__INFLUX__MEASUREMENT": "sqm",
            "SEEINGMON_SQM__INFLUX__FIELD": "mag",
            "SEEINGMON_SQM__INFLUX__TAGS__UNIT": "7",
            "SEEINGMON_SQM__INFLUX__TIMEOUT_S": "3",
        }
        sqm = self.load(tmp_path, "", env).section("sqm", SqmConfig)
        assert sqm.influx is not None
        assert sqm.influx.bucket == "2026"
        assert sqm.influx.tags == {"unit": "7"}
        assert sqm.influx.timeout_s == 3
        assert sqm.influx.token is not None
        assert sqm.influx.token.get_secret_value() == "12345"

    def test_the_effective_configuration_redacts_the_token_and_the_password(
        self, tmp_path: Path
    ) -> None:
        """The run record stores `Config.effective`, which hides a key by its name at any depth."""
        config = self.load(
            tmp_path,
            f'[sqm]\nenabled = true\nsource = "influx"\n[sqm.influx]\nendpoint = "{ENDPOINT}"\n'
            f'token = "{TOKEN}"\npassword = "{CODE}"\ntoken_env = "ENV_NAME_OF_TOKEN"\n'
            'password_env = "ENV_NAME_OF_CODE"\n'  # pragma: allowlist secret
            'measurement = "sqm"\nfield = "mag"\n',
        )
        effective = config.effective()
        table = effective["sqm"]["influx"]
        for key in ("token", "password", "token_env", "password_env", "endpoint"):
            assert table[key] == "<redacted>", key
        text = str(effective)
        for value in (TOKEN, CODE, ENDPOINT, "ENV_NAME_OF_TOKEN", "ENV_NAME_OF_CODE"):
            assert value not in text
        assert table["measurement"] == "sqm"  # the key names of the data are not secrets
