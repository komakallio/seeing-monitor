"""The building blocks of the configuration: merging, environment variables, and redaction."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pytest

from seeingmon.config import REDACTED, ConfigError
from seeingmon.config.layers import (
    deep_merge,
    env_overrides,
    is_deployment_key,
    is_secret_key,
    jsonable,
    parse_env_value,
    read_toml,
    redact,
)

# --- Merging ------------------------------------------------------------------------------------


def test_a_later_layer_overrides_an_earlier_value() -> None:
    assert deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_tables_merge_key_by_key_at_every_depth() -> None:
    base = {"site": {"a": 1, "inner": {"x": 1, "y": 2}}, "other": 1}
    override = {"site": {"b": 2, "inner": {"y": 3, "z": 4}}}
    assert deep_merge(base, override) == {
        "site": {"a": 1, "b": 2, "inner": {"x": 1, "y": 3, "z": 4}},
        "other": 1,
    }


def test_an_array_replaces_the_array_as_a_whole() -> None:
    assert deep_merge({"list": [1, 2, 3]}, {"list": [9]}) == {"list": [9]}
    assert deep_merge({"tables": [{"a": 1}]}, {"tables": [{"b": 2}]}) == {"tables": [{"b": 2}]}


def test_a_scalar_and_a_table_replace_each_other() -> None:
    assert deep_merge({"key": {"a": 1}}, {"key": 5}) == {"key": 5}
    assert deep_merge({"key": 5}, {"key": {"a": 1}}) == {"key": {"a": 1}}


def test_merging_changes_neither_input() -> None:
    base = {"a": {"b": [1, 2]}}
    override = {"a": {"c": {"d": 1}}}
    merged = deep_merge(base, override)
    merged["a"]["b"].append(3)
    merged["a"]["c"]["d"] = 99
    assert base == {"a": {"b": [1, 2]}}
    assert override == {"a": {"c": {"d": 1}}}


# --- Environment values -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("60", 60),
        ("-3", -3),
        ("1_000", 1000),
        ("0.5", 0.5),
        ("1e3", 1000.0),
        ("true", True),
        ("false", False),
        ("[1, 2, 3]", [1, 2, 3]),
        ('["a", "b"]', ["a", "b"]),
        ("[]", []),
        ('"12345"', "12345"),  # quotes force a string
        ("'literal'", "literal"),
        ("abc", "abc"),
        ("True", "True"),  # TOML booleans are lowercase
        ("", ""),
        ("two words", "two words"),
        ("1, 2", "1, 2"),  # an array needs brackets
        ("host:8086", "host:8086"),
        ("$argon2id$v=19$m=65536", "$argon2id$v=19$m=65536"),
        ("{a = 1}", "{a = 1}"),  # a table is not a scalar or an array
        ("1\nother = 2", "1\nother = 2"),  # a line break never starts another key
    ],
)
def test_an_environment_value_parses_as_toml_or_stays_a_string(text: str, expected: Any) -> None:
    result = parse_env_value(text)
    assert result == expected
    assert type(result) is type(expected)


def test_an_environment_value_can_be_a_toml_date() -> None:
    assert parse_env_value("2026-10-01") == date(2026, 10, 1)


# --- Environment variables ----------------------------------------------------------------------


def test_a_double_underscore_nests_and_a_single_underscore_stays_in_the_key() -> None:
    env = {
        "SEEINGMON_SITE__LATITUDE_DEG": "1.5",
        "SEEINGMON_SINKS__INFLUX__URL": "https://influx.example.com",
        "SEEINGMON_STATION_ID": "pi-1",
        "SEEINGMON_PROFILE": "other",
    }
    assert env_overrides(env) == {
        "site": {"latitude_deg": 1.5},
        "sinks": {"influx": {"url": "https://influx.example.com"}},
        "station_id": "pi-1",
        "profile": "other",
    }


def test_variables_without_the_prefix_are_ignored() -> None:
    env = {"PATH": "bin", "SEEINGMONITOR_X": "1", "seeingmon_x": "1", "EDITOR": "vi"}
    assert env_overrides(env) == {}


def test_names_are_split_after_the_prefix_only() -> None:
    assert env_overrides({"SEEINGMON_A1__B2_C": "x"}) == {"a1": {"b2_c": "x"}}


@pytest.mark.parametrize(
    "name",
    [
        "SEEINGMON_",
        "SEEINGMON__X",
        "SEEINGMON_X__",
        "SEEINGMON_X___Y",
        "SEEINGMON_x",
        "SEEINGMON_A-B",
    ],
)
def test_a_malformed_name_is_an_error(name: str) -> None:
    with pytest.raises(ConfigError, match="malformed environment variable"):
        env_overrides({name: "1"})


def test_the_error_for_a_malformed_name_never_shows_the_value() -> None:
    with pytest.raises(ConfigError) as error:
        env_overrides({"SEEINGMON_X__": "hunter2"})
    assert "hunter2" not in str(error.value)


def test_a_variable_that_sets_a_key_and_a_variable_that_nests_under_it_conflict() -> None:
    env = {"SEEINGMON_SITE": "5", "SEEINGMON_SITE__ELEVATION_M": "1"}
    with pytest.raises(
        ConfigError, match="SEEINGMON_SITE and SEEINGMON_SITE__ELEVATION_M conflict"
    ):
        env_overrides(env)


def test_two_names_for_one_key_cannot_both_be_set_because_names_are_unique() -> None:
    """`SEEINGMON_A_B` and `SEEINGMON_A__B` are different keys, so they do not conflict."""
    assert env_overrides({"SEEINGMON_A_B": "1", "SEEINGMON_A__B": "2"}) == {"a_b": 1, "a": {"b": 2}}


# --- Redaction ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "token",
        "token_hash",
        "api_token",
        "password",
        "db_password",
        "secret",
        "client_secret",
        "credential",
        "credentials",
        "key",
        "api_key",
        "private_key",
        "KEY",
        "Token",
        "PassWord",
        "monkey",  # a name that only contains the word is redacted too
    ],
)
def test_a_key_that_contains_a_secret_word_is_secret(key: str) -> None:
    assert is_secret_key(key)
    assert redact({key: "value"}) == {key: REDACTED}


@pytest.mark.parametrize("key", ["station_id", "latitude_deg", "user", "profile", "window_s"])
def test_other_keys_are_not_secret(key: str) -> None:
    assert not is_secret_key(key)
    assert redact({key: "value"}) == {key: "value"}


@pytest.mark.parametrize(
    "key",
    [
        "host",
        "url",
        "endpoint",
        "bind_address",
        "data_dir",
        "catalog_path",
        "power_command",
        "SOCKET",
        "log-file",
        "allowed_hosts",
        "extra_bind_addresses",
        "mirror_urls",
        "data_directories",
    ],
)
def test_a_key_that_names_a_deployment_value_is_redacted(key: str) -> None:
    assert is_deployment_key(key)
    assert redact({key: "value"}) == {key: REDACTED}


@pytest.mark.parametrize(
    "key",
    [
        "wind_direction_deg",
        "hostile",
        "profile",
        "pipeline_depth",
        "mode",
        "commands_per_window",
        "hostage_count",
    ],
)
def test_a_name_that_only_contains_a_deployment_word_is_not_redacted(key: str) -> None:
    assert not is_deployment_key(key)
    assert redact({key: 1}) == {key: 1}


def test_redaction_applies_at_any_depth_and_inside_arrays_of_tables() -> None:
    data = {
        "sinks": {"influx": {"url": "https://x.example.com", "Token": "abc"}},
        "list": [{"user": "u", "password": "p"}, {"user": "v"}],
        "plain": [1, 2, 3],
    }
    assert redact(data) == {
        "sinks": {"influx": {"url": REDACTED, "Token": REDACTED}},
        "list": [{"user": "u", "password": REDACTED}, {"user": "v"}],
        "plain": [1, 2, 3],
    }


def test_a_table_or_array_under_a_secret_key_is_replaced_as_a_whole() -> None:
    data = {"credentials": {"user": "u", "pass": "p"}, "api_keys": ["a", "b"]}
    assert redact(data) == {"credentials": REDACTED, "api_keys": REDACTED}


def test_redaction_replaces_values_of_any_type_and_keeps_none_of_them() -> None:
    data = {"token": 12345, "secret": True, "key": [1], "password": "", "credential": 1.5}
    assert set(redact(data).values()) == {REDACTED}


def test_redaction_does_not_change_its_input() -> None:
    data = {"auth": {"token_hash": "abc"}}
    redact(data)
    assert data == {"auth": {"token_hash": "abc"}}


def test_the_marker_is_fixed() -> None:
    assert REDACTED == "<redacted>"


# --- The keys that name the data of one installation ------------------------------------------


INFLUX_TABLE = {
    "endpoint": "https://influx.example.org:8086",
    "version": 2,
    "org": "an-org",
    "bucket": "a-bucket",
    "database": "a-database",
    "retention_policy": "a-policy",
    "username": "a-user",
    "measurement": "a-measurement",
    "field": "a-field",
    "temperature_field": "a-temperature-field",
    "tags": {"unit": "a-unit"},
    "timeout_s": 10.0,
    "max_age_s": 600.0,
}


def test_the_names_of_the_data_in_the_sqm_influx_table_are_redacted() -> None:
    hidden = {"endpoint", "org", "bucket", "database", "retention_policy", "username"}
    hidden |= {"measurement", "field", "temperature_field", "tags"}
    result = redact({"sqm": {"enabled": True, "influx": INFLUX_TABLE}})["sqm"]
    assert result["enabled"] is True
    for key, value in result["influx"].items():
        assert value == (REDACTED if key in hidden else INFLUX_TABLE[key]), key
    assert "a-unit" not in str(result)  # a table under a listed key goes as a whole


def test_the_same_key_names_in_other_tables_stay_as_they_are() -> None:
    data = {
        "sinks": {"lab": {"org": "an-org", "bucket": "a-bucket", "database": "a-database"}},
        "sqm": {"field": "not-in-the-influx-table", "tags": ["x"]},
        "influx": {"bucket": "a-bucket"},  # a table that is not at the path
        "survey": {"sqm": {"influx": {"bucket": "deeper-than-the-path"}}},
    }
    assert redact(data) == data


def test_an_array_of_tables_keeps_the_path_of_its_table() -> None:
    data = {"sqm": {"influx": INFLUX_TABLE, "other": [{"bucket": "kept"}]}}
    result = redact(data)["sqm"]
    assert result["other"] == [{"bucket": "kept"}]
    assert result["influx"]["bucket"] == REDACTED


# --- Plain JSON types -----------------------------------------------------------------------------


def test_toml_dates_and_times_become_iso_strings() -> None:
    data = {
        "when": datetime(2026, 10, 1, 12, 30, tzinfo=UTC),
        "day": date(2026, 10, 1),
        "at": time(6, 30),
        "nested": [{"d": date(2026, 1, 2)}],
        "n": 1,
    }
    result = jsonable(data)
    assert result == {
        "when": "2026-10-01T12:30:00+00:00",
        "day": "2026-10-01",
        "at": "06:30:00",
        "nested": [{"d": "2026-01-02"}],
        "n": 1,
    }
    json.dumps(result)  # serializes without a custom encoder


# --- Reading files --------------------------------------------------------------------------------


def test_a_toml_file_reads_into_a_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "a.toml"
    path.write_text('x = 1\n[t]\ny = "z"\n', encoding="utf-8")
    assert read_toml(path) == {"x": 1, "t": {"y": "z"}}


def test_a_file_that_is_not_toml_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("x = = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"broken\.toml: not valid TOML"):
        read_toml(path)


def test_a_file_that_is_not_utf8_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "binary.toml"
    path.write_bytes(b'x = "\xff"\n')
    with pytest.raises(ConfigError, match=r"binary\.toml: cannot read the file"):
        read_toml(path)


def test_a_missing_file_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"missing\.toml: cannot read the file"):
        read_toml(tmp_path / "missing.toml")
