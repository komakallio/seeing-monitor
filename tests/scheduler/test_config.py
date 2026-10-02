"""The scheduler configuration: the defaults file, the model, overrides, and the site."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from seeingmon.config import Config, ConfigError, load_config
from seeingmon.profile import load_profile
from seeingmon.scheduler.config import SchedulerConfig, SiteConfig, load_site, seconds_to_us

# A synthetic site: latitude 55 degrees north on the prime meridian. It is nobody's real site.
SYNTHETIC_SITE = {"latitude_deg": 55.0, "longitude_deg": 0.0}


@pytest.fixture
def defaults_file(repo_root: Path) -> Path:
    return repo_root / "config" / "default.d" / "scheduler.toml"


@pytest.fixture
def no_local(tmp_path: Path) -> Path:
    """A local file that does not exist, so a developer's own file stays out of the test."""
    return tmp_path / "missing.toml"


def read(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_the_defaults_file_keeps_every_key_under_the_scheduler_table(defaults_file: Path) -> None:
    assert set(read(defaults_file)) == {"scheduler"}


def test_the_defaults_file_and_the_models_agree(no_local: Path) -> None:
    """Both carry the defaults, so a change in one without the other fails here."""
    loaded = load_config(local_file=no_local, env={}).section("scheduler", SchedulerConfig)
    assert loaded == SchedulerConfig()


def test_the_defaults_file_sets_every_key_of_the_models(defaults_file: Path) -> None:
    """The file documents each key, so no model key may be missing from it."""
    table = read(defaults_file)["scheduler"]
    defaults = SchedulerConfig().model_dump()
    for section, values in defaults.items():
        assert set(table[section]) == set(values), section
    assert set(table) == set(defaults)


def test_the_defaults_state_that_they_are_provisional(defaults_file: Path) -> None:
    assert "PROVISIONAL" in defaults_file.read_text(encoding="utf-8")


def test_the_defaults_follow_the_architecture() -> None:
    config = SchedulerConfig()
    assert config.fast.window_s == 120.0
    assert config.fast.analysis_window_s == 60.0
    assert config.fast.exposure_us == 2000
    assert config.fast.high_speed is False
    assert config.fast.roi_arcmin == 4.1
    assert config.survey.cadence_s == 180.0
    assert config.watch.exposure_us == 1000
    assert config.watch.interval_s == 60.0
    assert config.daylight.saturation_limit == 0.5
    assert config.daylight.twilight_elevation_deg == -18.0
    assert config.align.idle_timeout_s == 30 * 60
    assert config.ladder.max_level == "power_cycle"


def test_the_survey_exposures_fit_the_cadence_and_the_profile() -> None:
    """A survey step needs less than the time that the fast period leaves in a cycle."""
    config = SchedulerConfig()
    step_s = config.survey.short_exposure_s + config.survey.long_exposure_s
    assert config.fast.window_s + step_s < config.survey.cadence_s
    assert config.cloud.fast_window_s + step_s < config.cloud.survey_cadence_s
    limits = load_profile("asi294mm-gs250").limits
    for exposure_us in (
        config.survey.short_exposure_us,
        config.survey.long_exposure_us,
        config.fast.exposure_us,
        config.watch.exposure_us,
    ):
        assert limits.exposure_us_range[0] <= exposure_us <= limits.exposure_us_range[1]
    for gain in (config.fast.gain, config.survey.short_gain, config.survey.long_gain):
        assert limits.gain_range[0] <= gain <= limits.gain_range[1]


def test_no_key_name_triggers_the_secret_redaction(no_local: Path) -> None:
    """`Config.effective` hides any key with `key`, `token`, and similar words in its name."""
    config = load_config(local_file=no_local, env={})
    assert config.effective()["scheduler"] == config.effective(redact=False)["scheduler"]


def test_an_environment_variable_overrides_a_nested_key(no_local: Path) -> None:
    env = {
        "SEEINGMON_SCHEDULER__FAST__WINDOW_S": "60",
        "SEEINGMON_SCHEDULER__WATCH__INTERVAL_S": "30",
    }
    config = load_config(local_file=no_local, env=env).section("scheduler", SchedulerConfig)
    assert config.fast.window_s == 60.0
    assert config.watch.interval_s == 30.0
    assert config.fast.analysis_window_s == 60.0  # the other keys keep their defaults


def test_the_local_file_overrides_a_default(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[scheduler.survey]\ncadence_s = 240.0\n", encoding="utf-8")
    config = load_config(local_file=local, env={}).section("scheduler", SchedulerConfig)
    assert config.survey.cadence_s == 240.0
    assert config.survey.long_gain == 120


def test_the_local_file_and_the_environment_turn_the_high_speed_mode_on(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[scheduler.fast]\nhigh_speed = true\n", encoding="utf-8")
    config = load_config(local_file=local, env={}).section("scheduler", SchedulerConfig)
    assert config.fast.high_speed is True
    env = {"SEEINGMON_SCHEDULER__FAST__HIGH_SPEED": "true"}
    config = load_config(local_file=tmp_path / "none.toml", env=env).section(
        "scheduler", SchedulerConfig
    )
    assert config.fast.high_speed is True


@pytest.mark.parametrize(
    "table",
    [
        {"fast": {"window_s": 30.0}},  # shorter than the analysis window
        {"fast": {"window_s": 0}},
        {"fast": {"exposure_us": 0}},
        {"fast": {"gain": -1}},
        {"fast": {"unknown_key": 1}},
        {"cloud": {"threshold": 0.3, "clear_threshold": 0.4}},
        {"cloud": {"threshold": 1.5}},
        {"daylight": {"saturation_limit": 0.3, "resume_saturation": 0.4}},
        {"daylight": {"sun_elevation_limit_deg": -20.0}},  # below the twilight limit
        {"faults": {"backoff_initial_s": 10.0, "backoff_max_s": 5.0}},
        {"faults": {"backoff_factor": 0.5}},
        {"ladder": {"max_level": "hammer"}},
        {"sweep": {"exposure_us": []}},
        {"nonexistent": {}},
    ],
    ids=lambda table: str(table),
)
def test_a_bad_table_is_rejected_with_the_section_named(table: dict[str, Any]) -> None:
    config = Config({"scheduler": table})
    with pytest.raises(ConfigError, match=r"\[scheduler\]"):
        config.section("scheduler", SchedulerConfig)


def test_the_models_are_frozen() -> None:
    config = SchedulerConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.fast.window_s = 1.0  # type: ignore[misc]


def test_exposures_convert_to_whole_microseconds() -> None:
    assert seconds_to_us(0.001) == 1000
    assert seconds_to_us(30.0) == 30_000_000
    assert seconds_to_us(1e-9) == 1  # never zero, because the camera rejects a zero exposure
    config = SchedulerConfig()
    assert config.survey.short_exposure_us == 1000
    assert config.survey.long_exposure_us == 30_000_000


class TestSite:
    def test_there_is_no_site_without_a_table(self) -> None:
        assert load_site(Config({})) is None

    def test_reads_the_latitude_and_the_longitude(self) -> None:
        site = load_site(Config({"site": {**SYNTHETIC_SITE, "elevation_m": 12.5}}))
        assert site == SiteConfig(latitude_deg=55.0, longitude_deg=0.0, elevation_m=12.5)

    def test_the_elevation_is_optional(self) -> None:
        site = load_site(Config({"site": SYNTHETIC_SITE}))
        assert site is not None
        assert site.elevation_m == 0.0

    def test_keys_that_other_parts_add_do_not_break_it(self) -> None:
        site = load_site(Config({"site": {**SYNTHETIC_SITE, "horizon_deg": 5.0}}))
        assert site is not None
        assert site.latitude_deg == 55.0

    @pytest.mark.parametrize(
        "table",
        [
            {"longitude_deg": 0.0},
            {"latitude_deg": 55.0},
            {"latitude_deg": 91.0, "longitude_deg": 0.0},
            {"latitude_deg": 55.0, "longitude_deg": 181.0},
            {"latitude_deg": "north", "longitude_deg": 0.0},
        ],
    )
    def test_a_bad_table_is_an_error_that_names_the_section(self, table: dict[str, Any]) -> None:
        with pytest.raises(ConfigError, match=r"\[site\]"):
            load_site(Config({"site": table}))

    def test_the_error_does_not_show_the_configured_value(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            load_site(Config({"site": {"latitude_deg": 123.456, "longitude_deg": 0.0}}))
        assert "123.456" not in str(excinfo.value)
