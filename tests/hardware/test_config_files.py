"""The defaults file and the local template of the hardware lane."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from seeingmon.config import load_config
from seeingmon.hardware.heater import HeaterConfig
from seeingmon.hardware.power import PowerConfig
from seeingmon.hardware.sqm import SqmConfig


@pytest.fixture
def defaults(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "config" / "default.d" / "hardware.toml").open("rb") as handle:
        return tomllib.load(handle)


def template_block(repo_root: Path) -> dict[str, Any]:
    """The commented TOML of the hardware part of the template, uncommented and parsed.

    A line counts as TOML when it is `# [table]` or `# key = value`. The prose lines of the
    template never have that shape.
    """
    text = (repo_root / "config" / "local.example.toml").read_text(encoding="utf-8")
    block = text[text.index("# Hardware.") :]
    toml_lines = []
    for line in block.splitlines():
        match = re.fullmatch(r"# (\[.*\]|[A-Za-z_][A-Za-z0-9_.]* = .*)", line)
        if match:
            toml_lines.append(match.group(1))
    return tomllib.loads("\n".join(toml_lines))


class TestDefaultsFile:
    def test_it_defines_only_the_three_parts_of_the_lane(self, defaults: dict[str, Any]) -> None:
        assert set(defaults) == {"heater", "sqm", "power"}

    def test_every_default_equals_the_default_of_its_model(self, tmp_path: Path) -> None:
        config = load_config(local_file=tmp_path / "none.toml", env={})
        assert config.section("heater", HeaterConfig) == HeaterConfig()
        assert config.section("sqm", SqmConfig) == SqmConfig()
        assert config.section("power", PowerConfig) == PowerConfig()

    def test_every_key_of_the_file_is_a_field_of_its_model(self, defaults: dict[str, Any]) -> None:
        for name, model in (("heater", HeaterConfig), ("sqm", SqmConfig), ("power", PowerConfig)):
            assert set(defaults[name]) <= set(model.model_fields), name

    def test_every_part_is_off_and_holds_no_deployment_value(
        self, defaults: dict[str, Any]
    ) -> None:
        assert defaults["heater"]["enabled"] is False
        assert defaults["sqm"]["enabled"] is False
        assert defaults["sqm"]["host"] == ""
        assert defaults["power"]["route"] == "none"
        assert "pins" not in defaults["heater"]

    def test_an_environment_variable_overrides_a_default(self, tmp_path: Path) -> None:
        config = load_config(
            local_file=tmp_path / "none.toml",
            env={"SEEINGMON_HEATER__MARGIN_C": "2.5", "SEEINGMON_SQM__PORT": "10002"},
        )
        assert config.section("heater", HeaterConfig).margin_c == 2.5
        assert config.section("sqm", SqmConfig).port == 10002


class TestTemplate:
    def test_the_commented_hardware_lines_form_valid_sections(self, repo_root: Path) -> None:
        values = template_block(repo_root)
        assert set(values) >= {"heater", "sqm", "power", "services"}
        heater = HeaterConfig.model_validate(values["heater"])
        assert heater.enabled
        assert heater.pins["heater"].chip.startswith("<")  # a placeholder, never a real pin map
        assert heater.ambient.kind == "sysfs"
        assert heater.optics.kind == "sysfs"
        sqm = SqmConfig.model_validate(values["sqm"])
        assert sqm.host.startswith("<")
        power = PowerConfig.model_validate(values["power"])
        assert power.route == "http"
        assert power.http.headers == {"Authorization": "Bearer ${PLUG_TOKEN}"}

    def test_the_template_names_the_camera_options_table_of_the_services_lane(
        self, repo_root: Path
    ) -> None:
        values = template_block(repo_root)
        assert values["services"]["acquire"]["driver"] == "asi"
        assert values["services"]["acquire"]["driver_options"]["library_path"].startswith("<")

    def test_every_value_that_belongs_to_one_installation_is_a_placeholder(
        self, repo_root: Path
    ) -> None:
        values = template_block(repo_root)
        placeholders = [
            values["heater"]["pins"]["heater"]["chip"],
            values["heater"]["ambient"]["temperature_file"],
            values["heater"]["optics"]["temperature_file"],
            values["sqm"]["host"],
            values["power"]["state_file"],
            values["power"]["command"]["argv"][0],
            values["services"]["acquire"]["driver_options"]["library_path"],
        ]
        for value in placeholders:
            assert value.startswith("<")
            assert value.endswith(">")
        assert "<" in values["power"]["http"]["url"]
        assert "${PLUG_TOKEN}" in values["power"]["http"]["headers"]["Authorization"]
