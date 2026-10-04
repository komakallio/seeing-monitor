"""The `[survey]` configuration: the defaults file matches the model, and overrides work."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import BaseModel

from seeingmon.config import ConfigError, load_config
from seeingmon.survey.config import SurveyConfig


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
    assert section.pointing.validity_s == 43_200.0
    assert section.fit.match_radius_px == (4.0, 2.0, 1.2)


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


def test_a_misspelled_key_is_an_error_that_names_it(tmp_path: Path) -> None:
    local = tmp_path / "config.toml"
    local.write_text("[survey.detect]\nthreshold_sigmaa = 3.0\n", encoding="utf-8")
    config = load_config(local_file=local, env={})
    with pytest.raises(ConfigError, match="threshold_sigmaa"):
        config.section("survey", SurveyConfig)
