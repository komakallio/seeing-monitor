"""The `[services]` configuration section and its defaults."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from seeingmon.clock import SystemClock
from seeingmon.config import REDACTED, ConfigError, load_config
from seeingmon.services.config import AcquireSettings, ClockSettings, ServicesConfig
from seeingmon.services.ipc.endpoint import PIPE_PREFIX
from seeingmon.services.ipc.errors import IpcConfigError
from seeingmon.services.ipc.keys import ConnectionKey

KEY = "a-test-key-with-32-characters-long"


def load(tmp_path: Path, env: dict[str, str] | None = None) -> ServicesConfig:
    config = load_config(local_file=tmp_path / "none.toml", env=env or {})
    return config.section("services", ServicesConfig)


def test_the_defaults_load_and_name_a_driver(tmp_path: Path) -> None:
    services = load(tmp_path)
    assert services.acquire.driver == "sim"
    assert services.acquire.queue_depth == 256
    assert services.acquire.gap_factor == 1.5
    assert services.connection_key is None
    assert services.clock.kind == "system"
    assert services.acquire.call_timeouts.read_grace_s > 0


def test_an_environment_variable_overrides_a_nested_key(tmp_path: Path) -> None:
    services = load(
        tmp_path,
        {
            "SEEINGMON_SERVICES__ACQUIRE__DRIVER": "fake",
            "SEEINGMON_SERVICES__ACQUIRE__QUEUE_DEPTH": "8",
            "SEEINGMON_SERVICES__ACQUIRE__DRIVER_OPTIONS__ADC_BITS": "12",
            "SEEINGMON_SERVICES__ACQUIRE__CALL_TIMEOUTS__OPEN_S": "2.5",
        },
    )
    assert services.acquire.driver == "fake"
    assert services.acquire.queue_depth == 8
    assert services.acquire.driver_options == {"adc_bits": 12}
    assert services.acquire.call_timeouts.open_s == 2.5


def test_a_misspelled_key_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="queue_dept"):
        load(tmp_path, {"SEEINGMON_SERVICES__ACQUIRE__QUEUE_DEPT": "8"})


def test_a_value_out_of_range_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="queue_depth"):
        load(tmp_path, {"SEEINGMON_SERVICES__ACQUIRE__QUEUE_DEPTH": "0"})


def test_the_warmup_cannot_exceed_the_fit_window() -> None:
    with pytest.raises(ValueError, match="fit_warmup"):
        AcquireSettings(fit_window=16, fit_warmup=32)


def test_the_key_comes_from_the_environment_and_stays_hidden(tmp_path: Path) -> None:
    config = load_config(
        local_file=tmp_path / "none.toml", env={"SEEINGMON_SERVICES__CONNECTION_KEY": KEY}
    )
    services = config.section("services", ServicesConfig)
    assert services.load_key(env={}) == ConnectionKey.from_text(KEY)
    assert KEY not in repr(services)
    assert KEY not in str(config.effective())
    assert config.effective()["services"]["connection_key"] == REDACTED


def test_an_all_digit_key_survives_the_environment_parser(tmp_path: Path) -> None:
    digits = "1234567890123456789012345678901234567890"
    services = load(tmp_path, {"SEEINGMON_SERVICES__CONNECTION_KEY": digits})
    assert services.load_key(env={}) == ConnectionKey.from_text(digits)


def test_a_missing_key_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(IpcConfigError, match="no connection key"):
        load(tmp_path).load_key(env={})


def test_the_key_file_and_credential_settings_reach_the_loader(tmp_path: Path) -> None:
    (tmp_path / "from-credential").write_text(KEY)
    services = load(
        tmp_path, {"SEEINGMON_SERVICES__CONNECTION_KEY_CREDENTIAL": '"from-credential"'}
    )
    key = services.load_key(env={"CREDENTIALS_DIRECTORY": str(tmp_path)})
    assert key == ConnectionKey.from_text(KEY)


def test_endpoints_default_by_platform_and_follow_the_setting(tmp_path: Path) -> None:
    services = load(tmp_path)
    assert services.endpoint("core", platform="win32").address == PIPE_PREFIX + "seeingmon-core"
    assert str(services.endpoint("acquire", platform="linux", env={})).endswith("acquire.sock")
    custom = load(tmp_path, {"SEEINGMON_SERVICES__ACQUIRE_ADDRESS": '"/x/a.sock"'})
    assert custom.endpoint("acquire", platform="linux").address == "/x/a.sock"
    assert str(custom.endpoint("core", platform="linux", env={})).endswith("core.sock")


class TestClockSettings:
    def test_the_default_is_the_system_clock(self) -> None:
        assert isinstance(ClockSettings().build(), SystemClock)

    def test_a_scaled_clock_needs_a_shared_origin(self) -> None:
        with pytest.raises(ValueError, match="origin_real_ns"):
            ClockSettings(kind="scaled", speed=10.0)

    def test_a_scaled_clock_runs_at_its_speed(self) -> None:
        settings = ClockSettings(
            kind="scaled", speed=50.0, start_utc_ns=1_800_000_000_000_000_000, origin_real_ns=0
        )
        clock = settings.build()
        assert clock.utc_ns() > 1_800_000_000_000_000_000
        assert clock.status().synchronized is True


def test_the_default_file_belongs_to_this_lane_and_parses(repo_root: Path) -> None:
    with (repo_root / "config" / "default.d" / "services.toml").open("rb") as handle:
        table = tomllib.load(handle)
    assert set(table) == {"services"}
