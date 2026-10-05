"""The `[survey]` configuration: the defaults file matches the model, and overrides work."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import BaseModel

from seeingmon.config import ConfigError, load_config
from seeingmon.survey.config import SurveyConfig, TransparencyConfig
from seeingmon.survey.detect import DetectOptions
from seeingmon.survey.quality import QualityOptions
from seeingmon.survey.transparency import TransparencyOptions


def read_defaults(repo_root: Path) -> dict[str, object]:
    with (repo_root / "config" / "default.d" / "survey.toml").open("rb") as handle:
        return dict(tomllib.load(handle)["survey"])


def test_the_defaults_file_gives_exactly_the_model_defaults(repo_root: Path) -> None:
    """The file documents the defaults, so it must not drift from the model."""
    from_file = SurveyConfig.model_validate(read_defaults(repo_root))
    assert from_file == SurveyConfig()


def test_the_defaults_file_names_every_key_of_the_model(repo_root: Path) -> None:
    defaults = read_defaults(repo_root)

    def keys(model_type: type[BaseModel], data: dict[str, object], path: str) -> None:
        for name, field in model_type.model_fields.items():
            assert name in data, f"{path}{name} is missing from config/default.d/survey.toml"
            nested = field.annotation
            if isinstance(nested, type) and hasattr(nested, "model_fields"):
                sub = data[name]
                assert isinstance(sub, dict)
                keys(nested, sub, f"{path}{name}.")

    keys(SurveyConfig, defaults, "survey.")


def test_the_configuration_layers_load_the_survey_section(tmp_path: Path) -> None:
    config = load_config(local_file=tmp_path / "missing.toml", env={})
    section = config.section("survey", SurveyConfig)
    assert section.catalog_path == ""
    assert section.solvers == ("astrometry.net", "astap")
    assert section.pointing.validity_s == 0.0  # no age limit
    assert section.fit.match_radius_px == (4.0, 2.0, 1.2)
    assert section.transparency.fallback_hours == 6.0


def test_the_local_file_and_the_environment_override_the_defaults(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text(
        '[survey]\ncatalog_path = "cap.smcat"\n\n[survey.solve]\ntimeout_s = 45.0\n',
        encoding="utf-8",
    )
    config = load_config(local_file=local, env={"SEEINGMON_SURVEY__POINTING__MOVED_ARCMIN": "7.5"})
    section = config.section("survey", SurveyConfig)
    assert section.catalog_path == "cap.smcat"
    assert section.solve.timeout_s == 45.0
    assert section.solve.max_stars == 1000  # a key that no layer overrides keeps its default
    assert section.solve.pole_hint_radius_deg == 15.0
    assert section.pointing.moved_arcmin == 7.5


def test_the_fallback_hours_reach_the_options_of_the_quality_step(tmp_path: Path) -> None:
    assert QualityOptions.from_config(SurveyConfig()).transparency.fallback_hours == 6.0
    local = tmp_path / "config.toml"
    local.write_text("[survey.transparency]\nfallback_hours = 2.5\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    quality = QualityOptions.from_config(config.section("survey", SurveyConfig))
    assert quality.transparency.fallback_hours == 2.5
    off = load_config(
        local_file=tmp_path / "missing.toml",
        env={"SEEINGMON_SURVEY__TRANSPARENCY__FALLBACK_HOURS": "0"},
    )
    assert QualityOptions.from_config(off.section("survey", SurveyConfig)).transparency == (
        TransparencyOptions(fallback_hours=0.0)
    )


def test_the_binned_search_is_on_by_default_and_a_local_file_turns_it_off(tmp_path: Path) -> None:
    default = load_config(local_file=tmp_path / "missing.toml", env={})
    options = DetectOptions.from_config(default.section("survey", SurveyConfig).detect)
    assert (options.coarse_bin, options.refine_stars) == (2, 1200)
    local = tmp_path / "config.toml"
    local.write_text("[survey.detect]\ncoarse_bin = 1\nrefine_stars = 900\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    options = DetectOptions.from_config(config.section("survey", SurveyConfig).detect)
    assert (options.coarse_bin, options.refine_stars) == (1, 900)
    env = load_config(
        local_file=tmp_path / "missing.toml", env={"SEEINGMON_SURVEY__DETECT__COARSE_BIN": "3"}
    )
    assert env.section("survey", SurveyConfig).detect.coarse_bin == 3


def test_a_bin_below_one_is_an_error_when_the_options_are_built(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[survey.detect]\ncoarse_bin = 0\n", encoding="utf-8")
    section = load_config(local_file=local, env={}).section("survey", SurveyConfig)
    with pytest.raises(ValueError, match="invalid detector options"):
        DetectOptions.from_config(section.detect)


def test_a_negative_fallback_is_an_error_when_the_options_are_built() -> None:
    config = SurveyConfig(transparency=TransparencyConfig(fallback_hours=-1.0))
    with pytest.raises(ValueError, match="invalid transparency options"):
        QualityOptions.from_config(config)


def test_a_misspelled_key_is_an_error_that_names_it(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[survey.detect]\nthreshold_sigmaa = 3.0\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError, match="threshold_sigmaa"):
        config.section("survey", SurveyConfig)


@pytest.mark.parametrize("value", ["-1.0", "nan", "inf"])
def test_a_validity_limit_below_zero_or_not_finite_fails_at_load(
    tmp_path: Path, value: str
) -> None:
    """0 means no age limit, and a bad value must not reach the tracker of every frame."""
    local = tmp_path / "config.toml"
    local.write_text(f"[survey.pointing]\nvalidity_s = {value}\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError, match="validity_s"):
        config.section("survey", SurveyConfig)
    default = load_config(local_file=tmp_path / "missing.toml", env={})
    assert default.section("survey", SurveyConfig).pointing.validity_s == 0.0
    limited = load_config(
        local_file=tmp_path / "missing.toml", env={"SEEINGMON_SURVEY__POINTING__VALIDITY_S": "3600"}
    )
    assert limited.section("survey", SurveyConfig).pointing.validity_s == 3600.0


@pytest.mark.parametrize("value", ["0.3", "0.2"])
def test_a_saturation_guard_at_or_below_the_target_fails_at_load(
    tmp_path: Path, value: str
) -> None:
    """A long frame at the target would get `saturated_sky`, and the scheduler would reject it."""
    local = tmp_path / "config.toml"
    local.write_text(f"[survey.twilight]\nmax_background_fraction = {value}\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError, match="max_background_fraction must lie above"):
        config.section("survey", SurveyConfig)
