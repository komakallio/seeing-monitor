"""`load_config` merges the layers, and `Config` reads, validates, and redacts the result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from seeingmon.config import REDACTED, Config, ConfigError, SectionModel, load_config
from seeingmon.profile import ProfileError


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A configuration folder with a `default.toml` and no `default.d/`."""
    directory = tmp_path / "config"
    write(directory / "default.toml", 'profile = "asi294mm-gs250"\nstation_id = "unset"\n')
    return directory


@pytest.fixture
def no_local(tmp_path: Path) -> Path:
    """A local-file path that does not exist, so the local layer is empty."""
    return tmp_path / "no-local" / "config.toml"


def load(
    config_dir: Path,
    no_local: Path,
    *,
    local_file: Path | None = None,
    env: dict[str, str] | None = None,
    profiles_dir: Path | None = None,
) -> Config:
    """Load with a custom folder, no local file, and an empty environment unless overridden."""
    return load_config(
        config_dir=config_dir,
        local_file=no_local if local_file is None else local_file,
        env={} if env is None else env,
        profiles_dir=profiles_dir,
    )


# --- Layers -----------------------------------------------------------------------------------


def test_the_layers_override_in_order(tmp_path: Path, config_dir: Path) -> None:
    write(config_dir / "default.d" / "a.toml", "[s]\nfrom_a = 1\nshared = 'a'\n")
    local = write(tmp_path / "local.toml", "[s]\nshared = 'local'\nlocal_only = 2\n")
    env = {"SEEINGMON_S__SHARED": "env", "SEEINGMON_S__ENV_ONLY": "3"}
    config = load_config(config_dir=config_dir, local_file=local, env=env)
    assert config.effective()["s"] == {
        "from_a": 1,
        "shared": "env",
        "local_only": 2,
        "env_only": 3,
    }


def test_the_local_file_overrides_the_defaults(
    tmp_path: Path, config_dir: Path, no_local: Path
) -> None:
    local = write(tmp_path / "local.toml", 'station_id = "pi-1"\n')
    assert load(config_dir, no_local, local_file=local).station_id == "pi-1"
    assert load(config_dir, no_local).station_id == "unset"


def test_the_environment_overrides_the_local_file(tmp_path: Path, config_dir: Path) -> None:
    local = write(tmp_path / "local.toml", 'station_id = "from-local"\n')
    config = load_config(config_dir=config_dir, local_file=local, env={"SEEINGMON_STATION_ID": "e"})
    assert config.station_id == "e"


def test_the_default_files_merge_in_sorted_file_name_order(
    config_dir: Path, no_local: Path
) -> None:
    defaults = config_dir / "default.d"
    write(defaults / "20-late.toml", "[s]\nvalue = 'late'\nlate = true\n")
    write(defaults / "05-early.toml", "[s]\nvalue = 'early'\nearly = true\n")
    write(defaults / "10-middle.toml", "[s]\nvalue = 'middle'\n")
    config = load(config_dir, no_local)
    assert config.effective()["s"] == {"value": "late", "late": True, "early": True}


def test_sorting_is_by_file_name_and_not_by_number(config_dir: Path, no_local: Path) -> None:
    """The name `10.toml` sorts before `2.toml`, so use zero-padded prefixes."""
    defaults = config_dir / "default.d"
    write(defaults / "2.toml", "value = 'two'\n")
    write(defaults / "10.toml", "value = 'ten'\n")
    assert load(config_dir, no_local).effective()["value"] == "two"


def test_the_default_d_folder_is_optional(config_dir: Path, no_local: Path) -> None:
    assert not (config_dir / "default.d").exists()
    assert load(config_dir, no_local).station_id == "unset"


def test_hidden_and_other_files_in_default_d_are_ignored(config_dir: Path, no_local: Path) -> None:
    defaults = config_dir / "default.d"
    write(defaults / ".hidden.toml", "ghost = 1\n")
    write(defaults / "notes.txt", "ghost = 1\n")
    write(defaults / "real.toml", "real = 1\n")
    (defaults / "folder.toml").mkdir()
    data = load(config_dir, no_local).effective()
    assert data["real"] == 1
    assert "ghost" not in data


def test_a_missing_default_file_is_an_error(tmp_path: Path, no_local: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigError, match="missing default configuration file"):
        load_config(config_dir=empty, local_file=no_local, env={})


def test_a_missing_local_file_is_not_an_error(config_dir: Path, no_local: Path) -> None:
    assert not no_local.exists()
    assert load(config_dir, no_local).profile_name == "asi294mm-gs250"


def test_a_broken_file_names_itself(config_dir: Path, no_local: Path) -> None:
    write(config_dir / "default.d" / "broken.toml", "x = = 1\n")
    with pytest.raises(ConfigError, match=r"broken\.toml: not valid TOML"):
        load(config_dir, no_local)


def test_a_malformed_environment_variable_is_an_error(config_dir: Path, no_local: Path) -> None:
    with pytest.raises(ConfigError, match="malformed environment variable"):
        load(config_dir, no_local, env={"SEEINGMON_X__": "1"})


def test_the_process_environment_is_the_default(
    config_dir: Path, no_local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEEINGMON_STATION_ID", "from-process-env")
    config = load_config(config_dir=config_dir, local_file=no_local)
    assert config.station_id == "from-process-env"


def test_the_default_directories_are_the_repository_ones(repo_root: Path, no_local: Path) -> None:
    config = load_config(local_file=no_local, env={})
    assert config.profile_name == "asi294mm-gs250"
    assert config.profile.id == "asi294mm-gs250"
    same = load_config(config_dir=repo_root / "config", local_file=no_local, env={})
    assert same.effective() == config.effective()


# --- Top-level keys -----------------------------------------------------------------------------


def test_the_station_id_and_the_profile_name_come_from_the_top_level(
    config_dir: Path, no_local: Path
) -> None:
    config = load(
        config_dir, no_local, env={"SEEINGMON_PROFILE": "other", "SEEINGMON_STATION_ID": "s-1"}
    )
    assert config.profile_name == "other"
    assert config.station_id == "s-1"


@pytest.mark.parametrize("bad", ["", "has space", "-leading", "a/b", "x" * 65])
def test_an_invalid_station_id_is_rejected_at_load_time(
    config_dir: Path, no_local: Path, bad: str
) -> None:
    with pytest.raises(ConfigError, match="station_id must be 1 to 64"):
        load(config_dir, no_local, env={"SEEINGMON_STATION_ID": f'"{bad}"'})


def test_a_station_id_that_is_not_a_string_is_rejected(config_dir: Path, no_local: Path) -> None:
    with pytest.raises(ConfigError, match="station_id must be"):
        load(config_dir, no_local, env={"SEEINGMON_STATION_ID": "42"})


def test_a_missing_station_id_is_reported_when_it_is_read(tmp_path: Path, no_local: Path) -> None:
    write(tmp_path / "bare" / "default.toml", "")
    config = load_config(config_dir=tmp_path / "bare", local_file=no_local, env={})
    with pytest.raises(ConfigError, match="station_id is not set"):
        assert config.station_id
    with pytest.raises(ConfigError, match="profile is not set"):
        assert config.profile_name


def test_the_configured_profile_loads(repo_root: Path, config_dir: Path, no_local: Path) -> None:
    config = load(config_dir, no_local, profiles_dir=repo_root / "profiles")
    assert config.profile.id == "asi294mm-gs250"
    assert config.profile is config.profile  # loaded once


def test_a_profile_name_can_be_a_path_to_a_file(
    repo_root: Path, tmp_path: Path, config_dir: Path, no_local: Path
) -> None:
    text = (repo_root / "profiles" / "asi294mm-gs250.toml").read_text(encoding="utf-8")
    path = write(
        tmp_path / "mine" / "my-camera.toml",
        text.replace('id = "asi294mm-gs250"', 'id = "my-camera"'),
    )
    config = load(config_dir, no_local, env={"SEEINGMON_PROFILE": f"'{path.as_posix()}'"})
    assert config.profile.id == "my-camera"


def test_an_unknown_profile_is_a_profile_error(
    repo_root: Path, config_dir: Path, no_local: Path
) -> None:
    config = load(
        config_dir, no_local, env={"SEEINGMON_PROFILE": "nope"}, profiles_dir=repo_root / "profiles"
    )
    with pytest.raises(ProfileError, match="no profile named 'nope'"):
        assert config.profile.id


# --- Sections -----------------------------------------------------------------------------------


class SchedulerConfig(SectionModel):
    window_s: float = Field(default=120.0, gt=0)
    cadence_s: int = 180
    states: list[str] = Field(default_factory=lambda: ["safe", "auto"])


class RequiredConfig(SectionModel):
    data_dir: str
    retries: int = 3


def test_a_section_validates_with_the_lanes_model(config_dir: Path, no_local: Path) -> None:
    write(config_dir / "default.d" / "scheduler.toml", "[scheduler]\nwindow_s = 60\n")
    section = load(config_dir, no_local).section("scheduler", SchedulerConfig)
    assert section == SchedulerConfig(window_s=60.0)
    assert section.window_s == 60.0
    assert isinstance(section.window_s, float)  # the model coerces the TOML integer


def test_the_models_defaults_apply_to_a_missing_key_and_a_missing_section(
    config_dir: Path, no_local: Path
) -> None:
    write(config_dir / "default.d" / "scheduler.toml", "[scheduler]\nwindow_s = 60\n")
    config = load(config_dir, no_local)
    assert config.section("scheduler", SchedulerConfig).cadence_s == 180
    assert config.section("not_there", SchedulerConfig) == SchedulerConfig()


def test_the_environment_can_override_a_section_key(config_dir: Path, no_local: Path) -> None:
    write(config_dir / "default.d" / "scheduler.toml", "[scheduler]\nwindow_s = 60\n")
    env = {"SEEINGMON_SCHEDULER__WINDOW_S": "30.5", "SEEINGMON_SCHEDULER__STATES": '["auto"]'}
    section = load(config_dir, no_local, env=env).section("scheduler", SchedulerConfig)
    assert section.window_s == 30.5
    assert section.states == ["auto"]


def test_a_dotted_name_reads_a_nested_table(config_dir: Path, no_local: Path) -> None:
    write(config_dir / "default.d" / "sinks.toml", "[sinks.influx]\nwindow_s = 5\n")
    section = load(config_dir, no_local).section("sinks.influx", SchedulerConfig)
    assert section.window_s == 5.0
    assert load(config_dir, no_local).section("sinks.absent", SchedulerConfig) == SchedulerConfig()


def test_an_invalid_value_names_the_section_and_the_key(config_dir: Path, no_local: Path) -> None:
    env = {"SEEINGMON_SCHEDULER__WINDOW_S": "-1"}
    with pytest.raises(ConfigError) as error:
        load(config_dir, no_local, env=env).section("scheduler", SchedulerConfig)
    assert str(error.value) == (
        "invalid configuration section [scheduler]:\n  window_s: Input should be greater than 0"
    )


def test_a_wrong_type_is_an_error(config_dir: Path, no_local: Path) -> None:
    env = {"SEEINGMON_SCHEDULER__CADENCE_S": "soon"}
    with pytest.raises(ConfigError, match="cadence_s: Input should be a valid integer"):
        load(config_dir, no_local, env=env).section("scheduler", SchedulerConfig)


def test_an_unknown_key_is_an_error_for_a_section_model(config_dir: Path, no_local: Path) -> None:
    env = {"SEEINGMON_SCHEDULER__WINDOW_SECONDS": "5"}
    with pytest.raises(ConfigError, match="window_seconds: Extra inputs are not permitted"):
        load(config_dir, no_local, env=env).section("scheduler", SchedulerConfig)


def test_a_plain_pydantic_model_works_too(config_dir: Path, no_local: Path) -> None:
    class Plain(BaseModel):
        value: int = 1

    assert load(config_dir, no_local).section("plain", Plain).value == 1


def test_a_missing_required_key_says_where_to_set_it(config_dir: Path, no_local: Path) -> None:
    with pytest.raises(ConfigError) as error:
        load(config_dir, no_local).section("paths", RequiredConfig)
    assert str(error.value) == (
        "invalid configuration section [paths]:\n"
        "  data_dir: Field required (set it in local/config.toml or in SEEINGMON_PATHS__DATA_DIR)"
    )


def test_several_problems_are_listed_together(config_dir: Path, no_local: Path) -> None:
    env = {"SEEINGMON_PATHS__RETRIES": "x"}
    with pytest.raises(ConfigError) as error:
        load(config_dir, no_local, env=env).section("paths", RequiredConfig)
    assert "data_dir: Field required" in str(error.value)
    assert "retries: Input should be a valid integer" in str(error.value)


def test_a_section_that_is_not_a_table_is_an_error(config_dir: Path, no_local: Path) -> None:
    with pytest.raises(ConfigError, match=r"configuration section \[scheduler\] must be a table"):
        load(config_dir, no_local, env={"SEEINGMON_SCHEDULER": "5"}).section(
            "scheduler", SchedulerConfig
        )


def test_a_section_error_never_shows_the_configured_value(config_dir: Path, no_local: Path) -> None:
    """A rejected secret must not reach a log through the error message or its traceback."""
    env = {"SEEINGMON_PATHS__RETRIES": "hunter2-not-a-number"}
    with pytest.raises(ConfigError) as error:
        load(config_dir, no_local, env=env).section("paths", RequiredConfig)
    assert "hunter2" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__


def test_a_section_model_is_frozen(config_dir: Path, no_local: Path) -> None:
    section = load(config_dir, no_local).section("scheduler", SchedulerConfig)
    with pytest.raises(ValueError, match="frozen"):
        section.window_s = 1.0  # type: ignore[misc]


def test_a_section_does_not_share_state_with_the_config(config_dir: Path, no_local: Path) -> None:
    write(config_dir / "default.d" / "scheduler.toml", '[scheduler]\nstates = ["a"]\n')
    config = load(config_dir, no_local)
    first = config.section("scheduler", SchedulerConfig)
    first.states.append("mutated")
    assert config.section("scheduler", SchedulerConfig).states == ["a"]


# --- The effective configuration ---------------------------------------------------------------


@pytest.fixture
def secret_config(config_dir: Path, tmp_path: Path) -> Config:
    local = write(
        tmp_path / "local.toml",
        "[site]\nlatitude_deg = 1.0\n"
        '[auth]\ntoken_hash = "abc"\n'
        '[sinks.influx]\nurl = "https://influx.example.com"\ntoken = "t"\n'
        '[[sinks.list]]\nuser = "u"\npassword = "p"\n'
        "[when]\nday = 2026-10-01\n",
    )
    return load_config(config_dir=config_dir, local_file=local, env={})


def test_the_effective_configuration_redacts_secrets_by_default(secret_config: Config) -> None:
    effective = secret_config.effective()
    assert effective["auth"] == {"token_hash": REDACTED}
    assert effective["sinks"]["influx"] == {"url": "https://influx.example.com", "token": REDACTED}
    assert effective["sinks"]["list"] == [{"user": "u", "password": REDACTED}]
    assert effective["site"] == {"latitude_deg": 1.0}
    assert effective["station_id"] == "unset"


def test_redaction_can_be_turned_off(secret_config: Config) -> None:
    effective = secret_config.effective(redact=False)
    assert effective["auth"] == {"token_hash": "abc"}
    assert effective["sinks"]["influx"]["token"] == "t"


def test_the_site_section_can_be_left_out(secret_config: Config) -> None:
    assert "site" in secret_config.effective()
    assert "site" not in secret_config.effective(omit_site=True)
    assert "site" not in secret_config.effective(redact=False, omit_site=True)


def test_the_effective_configuration_is_json(secret_config: Config) -> None:
    effective = secret_config.effective()
    assert effective["when"] == {"day": "2026-10-01"}
    assert json.loads(json.dumps(effective)) == effective


def test_the_effective_configuration_is_a_copy(secret_config: Config) -> None:
    secret_config.effective()["site"]["latitude_deg"] = 99.0
    assert secret_config.effective()["site"]["latitude_deg"] == 1.0


def test_the_representation_shows_section_names_and_no_values(secret_config: Config) -> None:
    text = repr(secret_config)
    assert "auth" in text
    assert "abc" not in text
    assert "latitude" not in text
