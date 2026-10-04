"""The `[services]` configuration section and its defaults."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from seeingmon.clock import SystemClock
from seeingmon.config import REDACTED, ConfigError, load_config
from seeingmon.services.config import AcquireSettings, ClockSettings, ServicesConfig
from seeingmon.services.core.settings import AlignmentSettings, CoreSettings
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
    assert set(table) == {"services", "alignment"}


class TestCoreSettings:
    def test_the_defaults_load_and_match_the_models(self, tmp_path: Path) -> None:
        config = load_config(local_file=tmp_path / "none.toml", env={})
        services = config.section("services", ServicesConfig)
        assert services.core == CoreSettings()
        assert services.core.survey_worker.mode == "process"
        assert services.core.escalation.reboot_command == []  # no reboot until you name one
        assert config.section("alignment", AlignmentSettings) == AlignmentSettings()

    def test_an_environment_variable_overrides_a_core_key(self, tmp_path: Path) -> None:
        services = load(
            tmp_path,
            {
                "SEEINGMON_SERVICES__CORE__HEALTH_INTERVAL_S": "5",
                "SEEINGMON_SERVICES__CORE__SURVEY_WORKER__MODE": '"thread"',
            },
        )
        assert services.core.health_interval_s == 5.0
        assert services.core.survey_worker.mode == "thread"

    def test_a_reboot_command_is_hidden_from_the_effective_configuration(
        self, tmp_path: Path
    ) -> None:
        local = tmp_path / "local.toml"
        local.write_text('[services.core.escalation]\nreboot_command = ["reboot-it", "now"]\n')
        config = load_config(local_file=local, env={})
        assert "reboot-it" not in str(config.effective())
        assert config.effective()["services"]["core"]["escalation"]["reboot_command"] == REDACTED

    def test_the_target_needs_both_coordinates(self) -> None:
        with pytest.raises(ValueError, match="together"):
            AlignmentSettings(target_x_px=10.0)
        assert not AlignmentSettings().has_target
        assert AlignmentSettings(target_x_px=1.0, target_y_px=2.0).has_target

    def test_the_quick_solve_runs_in_a_process_and_back_to_back_by_default(
        self, tmp_path: Path
    ) -> None:
        config = load_config(local_file=tmp_path / "none.toml", env={})
        settings = config.section("alignment", AlignmentSettings)
        assert settings.solver_mode == "process"  # the detector holds the GIL for seconds
        assert settings.solve_interval_s == 0.0  # the solver takes the newest frame when it is free
        assert settings.solve_timeout_s == 60.0
        assert (settings.detect_threshold_sigma, settings.detect_max_stars) == (8.0, 300)
        assert (settings.detect_coarse_bin, settings.detect_refine_stars) == (2, 300)

    def test_the_solver_mode_is_one_of_two(self, tmp_path: Path) -> None:
        local = tmp_path / "local.toml"
        local.write_text('[alignment]\nsolver_mode = "inline"\n')
        config = load_config(local_file=local, env={})
        with pytest.raises(ConfigError, match="solver_mode"):
            config.section("alignment", AlignmentSettings)

    def test_a_misspelled_alignment_key_fails_loudly(self, tmp_path: Path) -> None:
        local = tmp_path / "local.toml"
        local.write_text("[alignment]\ntarget_x = 1.0\n")
        config = load_config(local_file=local, env={})
        with pytest.raises(ConfigError, match="target_x"):
            config.section("alignment", AlignmentSettings)


class TestSurveyFrameSettings:
    """What `core` keeps of the survey frames (`[services.core.survey_frames]`)."""

    def test_the_defaults_follow_the_architecture(self, tmp_path: Path) -> None:
        settings = load(tmp_path).core.survey_frames
        assert settings.enabled is True
        assert settings.ram_frames == 2  # the short frame of a step waits in RAM for the long frame
        assert settings.keep_every == 10  # every tenth long frame goes to disk
        assert settings.preview_max_pixels == 1_000_000  # a preview of up to 1 megapixel
        assert settings.fits_compression == "rice"
        assert settings.event_min_interval_s == 3600.0
        assert settings.event_cloud_fraction == 0.5  # the threshold of the scheduler
        assert settings.event_background_fraction == 0.5  # the limit of the daylight gate

    def test_an_environment_variable_changes_a_key(self, tmp_path: Path) -> None:
        services = load(
            tmp_path,
            {
                "SEEINGMON_SERVICES__CORE__SURVEY_FRAMES__KEEP_EVERY": "5",
                "SEEINGMON_SERVICES__CORE__SURVEY_FRAMES__ENABLED": "false",
            },
        )
        assert services.core.survey_frames.keep_every == 5
        assert services.core.survey_frames.enabled is False

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("keep_every", "0"),
            ("ram_frames", "0"),
            ("ram_frames", "17"),
            ("jpeg_quality", "5"),
            ("jpeg_quality", "100"),
            ("preview_max_pixels", "100"),
            ("fits_compression", '"gzip"'),
            ("event_min_interval_s", "-1"),
            ("event_cloud_fraction", "1.5"),
            ("event_background_fraction", "0"),
            ("an_unknown_key", "1"),
        ],
    )
    def test_a_value_out_of_range_or_an_unknown_key_fails_loudly(
        self, tmp_path: Path, key: str, value: str
    ) -> None:
        local = tmp_path / "local.toml"
        local.write_text(f"[services.core.survey_frames]\n{key} = {value}\n")
        config = load_config(local_file=local, env={})
        with pytest.raises(ConfigError, match=key):
            config.section("services", ServicesConfig)
