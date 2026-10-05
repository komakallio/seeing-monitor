"""The `[fastpath]` configuration: the defaults file, the model, and the layers."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from seeingmon.config import ConfigError, load_config
from seeingmon.fastpath import FastPathConfig, create_fast_analyzer
from seeingmon.profile import Profile


def default_table(repo_root: Path) -> dict[str, object]:
    with (repo_root / "config" / "default.d" / "fastpath.toml").open("rb") as handle:
        table = tomllib.load(handle)["fastpath"]
    assert isinstance(table, dict)
    return table


def test_the_defaults_file_and_the_model_agree(repo_root: Path) -> None:
    """Every key of the file is a field, and every value equals the model's default."""
    table = default_table(repo_root)
    defaults = FastPathConfig()
    unknown = set(table) - set(FastPathConfig.model_fields)
    assert not unknown, f"keys that the model does not declare: {sorted(unknown)}"
    for key, value in table.items():
        default = getattr(defaults, key)
        # A TOML array reads as a list, and the model keeps a tuple.
        assert (list(default) if isinstance(default, tuple) else default) == value, key


def test_every_field_has_a_line_in_the_defaults_file_except_the_derived_aperture(
    repo_root: Path,
) -> None:
    missing = set(FastPathConfig.model_fields) - set(default_table(repo_root))
    assert missing == {"aperture_diameter_px"}


def test_the_section_reads_through_the_configuration(tmp_path: Path) -> None:
    config = load_config(local_file=tmp_path / "absent.toml", env={})
    section = config.section("fastpath", FastPathConfig)
    assert section == FastPathConfig()
    assert section.window_s == 60.0
    assert section.outer_scale_m == 20.0
    assert section.assumed_wind_ms == 10.0
    assert section.detrend_order == 2
    assert section.vibration_threshold == 5.0


def test_a_local_file_and_the_environment_override_the_defaults(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[fastpath]\nwindow_s = 30.0\nouter_scale_m = 25.0\n", encoding="utf-8")
    config = load_config(
        local_file=local,
        env={"SEEINGMON_FASTPATH__ASSUMED_WIND_MS": "6", "SEEINGMON_FASTPATH__OUTER_SCALE_M": "40"},
    )
    section = config.section("fastpath", FastPathConfig)
    assert section.window_s == 30.0  # from the local file
    assert section.outer_scale_m == 40.0  # the environment wins over the file
    assert section.assumed_wind_ms == 6.0


def test_a_misspelled_key_is_an_error(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[fastpath]\nwindow_seconds = 30\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError, match="window_seconds"):
        config.section("fastpath", FastPathConfig)


@pytest.mark.parametrize(
    "text",
    [
        "window_s = 0",
        "window_s = -5.0",
        "min_valid_fraction = 1.5",
        "detrend_order = 9",
        "outer_scale_m = -1.0",
        "vibration_threshold = 0.5",
        "welch_overlap = 1.0",
        "structure_lag_min_s = 0.5\nstructure_lag_max_s = 0.1",
        "window_s = 4.0\nmin_window_s = 5.0",
        "aperture_diameter_px = 2.0",
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, text: str) -> None:
    local = tmp_path / "config.toml"
    local.write_text(f"[fastpath]\n{text}\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError, match=r"fastpath"):
        config.section("fastpath", FastPathConfig)


def test_the_model_is_frozen_and_the_defaults_match_the_architecture() -> None:
    config = FastPathConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.window_s = 10.0  # type: ignore[misc]
    assert config.saturation_fraction == 0.98
    assert config.recenter_iterations == 2
    assert config.aperture_min_px == 15.0
    assert config.g_tilt_coefficient == 0.170
    assert config.fwhm_coefficient == 0.98


def test_create_fast_analyzer_uses_the_configured_window(profile: Profile) -> None:
    analyzer = create_fast_analyzer(profile, FastPathConfig(window_s=20.0), "station")
    assert analyzer is not None
    # The window length reaches the assembler through the configuration.
    assert analyzer._assembler.window_s == 20.0
