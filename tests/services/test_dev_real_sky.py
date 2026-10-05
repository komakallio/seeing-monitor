"""`seeingmon dev --driver asi --real-sky`: the plan of a run on the real sky, with nothing started.

The tests use made-up values only: a site that is nobody's, a small synthetic cap catalog that a
test writes (the launcher reads its header, and `core` reads it later), and solver commands that
name a stub file or no program at all. No test starts a child, opens a camera, or touches the
network. The plan only decides what the children receive, and the tests read that back the way
`core` does, through the configuration layers.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("fastapi", reason="the web settings come from the web extra")

from seeingmon.cli import CliError, main
from seeingmon.clock import iso_to_utc_ns
from seeingmon.config import load_config
from seeingmon.fastpath import FastPathConfig
from seeingmon.scheduler import SchedulerConfig, SiteConfig
from seeingmon.services.config import ServicesConfig
from seeingmon.services.core.settings import AlignmentSettings, CoreSettings
from seeingmon.services.dev import (
    SITE_KEYS,
    DevOptions,
    DevPlan,
    banner,
    build_plan,
    options_from_args,
    render_env_value,
    run_dev,
)
from seeingmon.survey.config import SurveyConfig
from tests.services.test_dev import FORBIDDEN, OWNER, child, seeingmon_env

SITE = {"latitude_deg": 12.5, "longitude_deg": 34.5, "elevation_m": 123.5}  # made up
SITE_NUMBERS = ("12.5", "34.5", "123.5")
# The values of the owner file of `test_dev` that a real-sky run must keep from every child. The
# site of that file (12.5 and 34.5) is a value that `core` receives in this mode, so it is not here.
FORBIDDEN_IN_A_REAL_SKY = tuple(text for text in FORBIDDEN if text not in ("12.5", "34.5"))
OWNER_WITHOUT_A_SITE = OWNER.replace("[site]\nlatitude_deg = 12.5\nlongitude_deg = 34.5\n", "")
NOT_THERE = "this-program-does-not-exist-example"


def toml(tables: dict[str, Any]) -> str:
    """The text of a local configuration with the given tables (nested tables get their own)."""
    lines: list[str] = []

    def emit(name: str, table: dict[str, Any]) -> None:
        values = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if values or not table:
            lines.append(f"[{name}]")
            lines.extend(f"{key} = {render_env_value(value)}" for key, value in values.items())
        for key, value in table.items():
            if isinstance(value, dict):
                emit(f"{name}.{key}", value)

    for name, table in tables.items():
        emit(name, table)
    return "\n".join(lines) + "\n"


def stub_program(directory: Path, name: str = "solver-stub.example") -> str:
    """A command that names a file that exists, written the way that a person writes it."""
    path = directory / name
    path.write_text("", encoding="utf-8")
    return f'"{path.as_posix()}"'  # quotes keep a path with a space in one piece


@functools.cache
def catalog_bytes() -> bytes:
    """A small cap catalog, made once. The launcher reads its header, and `core` loads it later."""
    from seeingmon.survey.catalog import write_catalog
    from tests.survey import synth

    with tempfile.TemporaryDirectory(prefix="smon-cat-") as folder:
        path = Path(folder) / "cap.example"
        write_catalog(path, synth.synthetic_catalog(cap_radius_deg=5.0, density_scale=0.01))
        return path.read_bytes()


def catalog_file(directory: Path) -> Path:
    path = directory / "cap.example"
    path.write_bytes(catalog_bytes())
    return path


def working_tables(directory: Path, **survey: Any) -> dict[str, Any]:
    """The site and survey tables of a person who set everything and whose solver is installed."""
    return {
        "site": dict(SITE),
        "survey": {
            "catalog_path": str(catalog_file(directory)),
            "solvers": ["astap"],
            "astap_command": stub_program(directory),
            **survey,
        },
    }


def real_plan(
    tmp_path: Path,
    tables: dict[str, Any] | None = None,
    *,
    text: str = "",
    env: dict[str, str] | None = None,
    name: str = "run",
    origin_real_ns: int | None = None,
    **options: Any,
) -> DevPlan:
    """The plan of a real-sky run. `text` comes before the tables in the local configuration."""
    local = tmp_path / f"{name}-owner.toml"
    shown = working_tables(tmp_path) if tables is None else tables
    local.write_text(text + toml(shown), encoding="utf-8")
    return build_plan(
        DevOptions(
            acquire_driver="asi", real_sky=True, **({"data_dir": tmp_path / "data"} | options)
        ),
        directory=tmp_path / name,
        local_file=local,
        env=env if env is not None else {},
        origin_real_ns=origin_real_ns,
    )


def configuration(plan: DevPlan, name: str, tmp_path: Path) -> Any:
    """What a child reads from its environment, as `seeingmon core` loads it."""
    return load_config(local_file=tmp_path / "absent.toml", env=seeingmon_env(child(plan, name)))


def lines_without_the_urls(plan: DevPlan) -> list[str]:
    return [line for line in banner(plan) if not line.startswith("Web UI:")]


class TestThePlanOfTheRealSky:
    def test_core_gets_the_site_the_survey_and_the_alignment_of_the_owner(
        self, tmp_path: Path
    ) -> None:
        tables = working_tables(
            tmp_path,
            index_dir=str(tmp_path / "index"),
            astap_database_dir=str(tmp_path / "astap-db"),
            hot_pixel_file=str(tmp_path / "hot.npy"),
        )
        tables["alignment"] = {"aim_x_px": 10.5, "aim_y_px": 20.5, "target_roll_deg": 1.5}
        tables["alignment"] |= {"target_x_px": 30.5, "target_y_px": 40.5}
        plan = real_plan(tmp_path, tables)
        core = configuration(plan, "core", tmp_path)
        site = core.section("site", SiteConfig)
        assert (site.latitude_deg, site.longitude_deg, site.elevation_m) == (12.5, 34.5, 123.5)
        survey = core.section("survey", SurveyConfig)
        assert survey.catalog_path == tables["survey"]["catalog_path"]
        assert survey.solvers == ("astap",)
        assert survey.astap_command == tables["survey"]["astap_command"]
        assert survey.index_dir == str(tmp_path / "index")
        assert survey.astap_database_dir == str(tmp_path / "astap-db")
        assert survey.hot_pixel_file == str(tmp_path / "hot.npy")
        alignment = core.section("alignment", AlignmentSettings)
        assert alignment.aim_xy == (10.5, 20.5)
        assert (alignment.target_x_px, alignment.target_y_px) == (30.5, 40.5)
        assert alignment.target_roll_deg == 1.5

    def test_the_solver_settings_keep_their_defaults_when_the_owner_sets_none(
        self, tmp_path: Path
    ) -> None:
        plan = real_plan(
            tmp_path, {"site": dict(SITE), "survey": working_tables(tmp_path)["survey"]}
        )
        survey = configuration(plan, "core", tmp_path).section("survey", SurveyConfig)
        assert survey.solve_field_command == "solve-field"
        assert survey.index_dir == ""
        assert survey.astap_database_dir == ""

    def test_nothing_is_seeded_and_no_sky_is_simulated(self, tmp_path: Path) -> None:
        plan = real_plan(tmp_path)
        assert plan.real
        assert plan.real_sky
        core = child(plan, "core").env
        assert "SEEINGMON_SERVICES__CORE__SEED_SOLUTION_FILE" not in core
        services = configuration(plan, "core", tmp_path).section("services", ServicesConfig)
        assert services.core.seed_solution_file == ""
        assert services.core == CoreSettings()  # the production settings of core, none of the dev
        assert plan.cover_file is None
        written = [path.name for path in plan.directory.rglob("*") if path.is_file()]
        assert written == []  # no catalog.bin, no seed.json, no small profile
        survey = configuration(plan, "core", tmp_path).section("survey", SurveyConfig)
        assert Path(survey.catalog_path).parent == tmp_path  # the catalog of the owner

    def test_the_profile_the_clock_and_the_data_folder(self, tmp_path: Path) -> None:
        plan = real_plan(tmp_path)
        for name in ("acquire", "core", "web"):
            config = configuration(plan, name, tmp_path)
            assert config.profile.id == "asi294mm-gs250"
            services = config.section("services", ServicesConfig)
            assert services.clock.kind == "system"
            assert config.station_id == "dev"
            env = child(plan, name).env
            assert Path(json.loads(env["SEEINGMON_PATHS__DATA_DIR"])) == tmp_path / "data"
        acquire = configuration(plan, "acquire", tmp_path).section("services", ServicesConfig)
        assert acquire.acquire.driver == "asi"
        assert acquire.acquire.driver_options == {}

    def test_the_windows_stay_20_s_long(self, tmp_path: Path) -> None:
        core = configuration(real_plan(tmp_path), "core", tmp_path)
        assert core.section("scheduler", SchedulerConfig).fast.analysis_window_s == 20.0
        # seven windows and the survey step fill the 180 s cadence
        assert core.section("scheduler", SchedulerConfig).fast.window_s == 140.0
        assert core.section("fastpath", FastPathConfig).window_s == 20.0

    def test_the_cloud_limits_are_the_production_defaults(self, tmp_path: Path) -> None:
        survey = configuration(real_plan(tmp_path), "core", tmp_path).section(
            "survey", SurveyConfig
        )
        assert survey.cloud == SurveyConfig().cloud  # not the 4, 10, and 13 of the synthetic sky
        assert (survey.cloud.min_expected, survey.cloud.expected_snr) == (8, 20.0)

    def test_the_owner_sets_the_cloud_limits_and_the_dark_session_in_the_survey_table(
        self, tmp_path: Path
    ) -> None:
        tables = working_tables(tmp_path)
        tables["survey"]["cloud"] = {"min_expected": 3}
        tables["survey"]["dark"] = {"frames": 7}
        survey = configuration(real_plan(tmp_path, tables), "core", tmp_path).section(
            "survey", SurveyConfig
        )
        assert survey.cloud.min_expected == 3
        assert survey.cloud.expected_snr == 20.0  # the rest of the table keeps its default
        assert (survey.dark.frames, survey.dark.bias_frames, survey.dark.poll_s) == (7, 5, 2.0)

    def test_the_owner_names_the_pointing_reference_in_the_pointing_table(
        self, tmp_path: Path
    ) -> None:
        """`seeingmon pointing set-reference` prints `reference_file` for `[survey.pointing]`."""
        reference = tmp_path / "calibration" / "pointing-reference.json"
        tables = working_tables(tmp_path)
        tables["survey"]["pointing"] = {"reference_file": str(reference), "moved_arcmin": 2.5}
        plan = real_plan(tmp_path, tables)
        survey = configuration(plan, "core", tmp_path).section("survey", SurveyConfig)
        assert survey.pointing.reference_file == str(reference)
        assert survey.pointing.moved_arcmin == 2.5
        assert survey.pointing.few_stars == 12  # the rest of the table keeps its default
        assert survey.pointing.validity_s == 0.0  # no age limit
        assert survey.catalog_path == tables["survey"]["catalog_path"]  # and so does the rest
        variable = "SEEINGMON_SURVEY__POINTING__REFERENCE_FILE"
        assert json.loads(child(plan, "core").env[variable]) == str(reference)
        for name in ("acquire", "web"):  # only core receives the survey table
            assert variable not in child(plan, name).env, name

    def test_a_variable_can_name_the_pointing_reference_and_beats_the_file(
        self, tmp_path: Path
    ) -> None:
        in_file = tmp_path / "from-the-file.json"
        in_variable = tmp_path / "from-the-variable.json"
        tables = working_tables(tmp_path)
        tables["survey"]["pointing"] = {"reference_file": str(in_file)}
        env = {"SEEINGMON_SURVEY__POINTING__REFERENCE_FILE": json.dumps(str(in_variable))}
        plan = real_plan(tmp_path, tables, env=env)
        survey = configuration(plan, "core", tmp_path).section("survey", SurveyConfig)
        assert survey.pointing.reference_file == str(in_variable)
        alone = real_plan(tmp_path, working_tables(tmp_path), env=env, name="alone")
        survey = configuration(alone, "core", tmp_path).section("survey", SurveyConfig)
        assert survey.pointing.reference_file == str(in_variable)

    def test_without_a_pointing_table_core_loads_no_reference(self, tmp_path: Path) -> None:
        survey = configuration(real_plan(tmp_path), "core", tmp_path).section(
            "survey", SurveyConfig
        )
        # With no file, the Pointing card says "no reference solution".
        assert survey.pointing.reference_file == ""

    def test_the_dark_session_is_short_unless_the_owner_says_otherwise(
        self, tmp_path: Path
    ) -> None:
        dark = (
            configuration(real_plan(tmp_path), "core", tmp_path)
            .section("survey", SurveyConfig)
            .dark
        )
        assert (dark.frames, dark.bias_frames, dark.poll_s) == (5, 5, 2.0)
        assert dark.exposure_s == 30.0  # the exposure of the survey stays

    def test_the_dark_library_is_in_the_data_folder_unless_the_owner_names_one(
        self, tmp_path: Path
    ) -> None:
        default = configuration(real_plan(tmp_path), "core", tmp_path).section(
            "survey", SurveyConfig
        )
        assert default.calibration_dir == str(tmp_path / "data" / "calibration")
        named = working_tables(tmp_path, calibration_dir=str(tmp_path / "elsewhere"))
        plan = real_plan(tmp_path, named, name="named")
        survey = configuration(plan, "core", tmp_path).section("survey", SurveyConfig)
        assert survey.calibration_dir == str(tmp_path / "elsewhere")
        blank = real_plan(tmp_path, working_tables(tmp_path, calibration_dir=""), name="blank")
        blank_survey = configuration(blank, "core", tmp_path).section("survey", SurveyConfig)
        assert blank_survey.calibration_dir == str(tmp_path / "data" / "calibration")

    def test_a_variable_beats_the_file_as_in_every_layer(self, tmp_path: Path) -> None:
        env = {"SEEINGMON_SITE__ELEVATION_M": "55.5", "SEEINGMON_SURVEY__SOLVERS": "[]"}
        plan = real_plan(tmp_path, env=env)
        core = configuration(plan, "core", tmp_path)
        assert core.section("site", SiteConfig).elevation_m == 55.5
        assert core.section("survey", SurveyConfig).solvers == ()

    def test_the_variables_alone_can_supply_the_tables(self, tmp_path: Path) -> None:
        env = {
            "SEEINGMON_SITE__LATITUDE_DEG": "12.5",
            "SEEINGMON_SITE__LONGITUDE_DEG": "34.5",
            "SEEINGMON_SITE__ELEVATION_M": "123.5",
            "SEEINGMON_SURVEY__CATALOG_PATH": json.dumps(str(catalog_file(tmp_path))),
            "SEEINGMON_SURVEY__SOLVERS": '["astap"]',
            "SEEINGMON_SURVEY__ASTAP_COMMAND": json.dumps(stub_program(tmp_path)),
            "SEEINGMON_ALIGNMENT__AIM_X_PX": "1.5",
            "SEEINGMON_ALIGNMENT__AIM_Y_PX": "2.5",
        }
        plan = real_plan(tmp_path, {}, env=env)
        core = configuration(plan, "core", tmp_path)
        assert core.section("site", SiteConfig).latitude_deg == 12.5
        assert core.section("alignment", AlignmentSettings).aim_xy == (1.5, 2.5)
        assert plan.warnings == []

    def test_the_simulated_plan_still_makes_its_own_sky(self, tmp_path: Path) -> None:
        local = tmp_path / "owner.toml"
        local.write_text(toml(working_tables(tmp_path)), encoding="utf-8")
        plan = build_plan(DevOptions(), directory=tmp_path / "sim", local_file=local, env={})
        assert not plan.real_sky
        core = child(plan, "core").env
        assert Path(json.loads(core["SEEINGMON_SERVICES__CORE__SEED_SOLUTION_FILE"])).is_file()
        assert float(core["SEEINGMON_SITE__LATITUDE_DEG"]) == 55.0  # the synthetic site
        assert plan.warnings == []
        assert plan.log_dir is None


class TestTheIsolationOfTheRealSky:
    def test_only_core_receives_the_three_tables(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path)
        tables["alignment"] = {"aim_x_px": 1.5, "aim_y_px": 2.5}
        plan = real_plan(tmp_path, tables)
        prefixes = ("SEEINGMON_SITE__", "SEEINGMON_SURVEY__", "SEEINGMON_ALIGNMENT__")
        for name in ("acquire", "web"):
            assert not [k for k in child(plan, name).env if k.startswith(prefixes)], name
        core = child(plan, "core").env
        for prefix in prefixes:
            assert [k for k in core if k.startswith(prefix)], prefix

    def test_nothing_else_of_the_owner_reaches_a_child(self, tmp_path: Path) -> None:
        plan = real_plan(tmp_path, text=OWNER_WITHOUT_A_SITE, tables=None)
        everything = json.dumps([[spec.argv, spec.env] for spec in plan.children], sort_keys=True)
        for text in FORBIDDEN_IN_A_REAL_SKY:
            assert text not in everything, text
        for path in tmp_path.rglob("*"):
            if path.is_file() and path.name != "run-owner.toml":  # the file of the owner itself
                for text in FORBIDDEN_IN_A_REAL_SKY:
                    assert text.encode() not in path.read_bytes(), (path.name, text)
        web = child(plan, "web").env
        assert json.loads(web["SEEINGMON_WEB__PORT"]) == 8123  # the web part is still the owner's
        assert json.loads(child(plan, "core").env["SEEINGMON_STATION_ID"]) == "dev"

    def test_a_variable_of_another_section_never_reaches_a_child(self, tmp_path: Path) -> None:
        env = {
            "SEEINGMON_SINKS__INFLUX__TOKEN": "another-owner-secret",  # pragma: allowlist secret
            "SEEINGMON_SERVICES__CONNECTION_KEY": "another-owner-key",  # pragma: allowlist secret
            "SEEINGMON_SCHEDULER__SEARCH__MAX_SUN_ELEVATION_DEG": "7",
            "SEEINGMON_HEATER__ENABLED": "true",
            "SEEINGMON_STATION_ID": '"another-station"',
        }
        plan = real_plan(tmp_path, env=env)
        everything = json.dumps([spec.env for spec in plan.children])
        for value in ("another-owner-secret", "another-owner-key", "another-station"):
            assert value not in everything, value
        for spec in plan.children:  # the names that the launcher does not set itself
            assert "SEEINGMON_SINKS__INFLUX__TOKEN" not in spec.env
            assert "SEEINGMON_HEATER__ENABLED" not in spec.env
            assert "SEEINGMON_SCHEDULER__SEARCH__MAX_SUN_ELEVATION_DEG" not in spec.env
        scheduler = configuration(plan, "core", tmp_path).section("scheduler", SchedulerConfig)
        assert scheduler.search.max_sun_elevation_deg == 90.0  # the production default

    def test_no_value_of_the_tables_goes_on_a_command_line(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path, index_dir=str(tmp_path / "index-folder"))
        tables["alignment"] = {"aim_x_px": 1.25, "aim_y_px": 2.25}
        plan = real_plan(tmp_path, tables)
        for spec in plan.children:
            line = " ".join(spec.argv)
            values = (*SITE_NUMBERS, "1.25", "2.25", "cap.example", "solver-stub", "index-folder")
            for text in values:
                assert text not in line, (spec.name, text)

    def test_the_run_folder_holds_no_value_of_the_owner(self, tmp_path: Path) -> None:
        plan = real_plan(tmp_path)
        for path in [*plan.directory.rglob("*"), *(tmp_path / "data").rglob("*")]:
            if path.is_file():
                for text in SITE_NUMBERS:
                    assert text.encode() not in path.read_bytes(), (path.name, text)


class TestTheChecks:
    """The launcher refuses what a real night cannot do, and names the table and the setting."""

    def refused(self, tmp_path: Path, tables: dict[str, Any], **options: Any) -> CliError:
        with pytest.raises(CliError) as raised:
            real_plan(tmp_path, tables, **options)
        assert raised.value.exit_code == 2
        assert not (tmp_path / "run").exists()  # nothing was written
        assert not (tmp_path / "data").exists()
        return raised.value

    @pytest.mark.parametrize("missing", [None, "latitude_deg", "longitude_deg", "elevation_m"])
    def test_a_site_that_lacks_a_value_is_refused_where_the_values_belong(
        self, tmp_path: Path, missing: str | None
    ) -> None:
        tables = working_tables(tmp_path)
        if missing is None:
            del tables["site"]
        else:
            del tables["site"][missing]
        text = str(self.refused(tmp_path, tables))
        assert "[site]" in text
        assert "latitude_deg, longitude_deg, elevation_m" in text  # what a real sky needs
        expected = list(SITE_KEYS) if missing is None else [missing]
        assert f"lacks {', '.join(expected)}." in text
        assert "untracked local/config.toml" in text
        assert "SEEINGMON_SITE__" in text
        assert not any(number in text for number in SITE_NUMBERS)  # no value in a message

    def test_the_placeholder_site_of_the_template_is_refused(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path)
        tables["site"] = {"latitude_deg": 0.0, "longitude_deg": 0.0, "elevation_m": 0.0}
        text = str(self.refused(tmp_path, tables))
        assert "[site]" in text
        assert "placeholders" in text
        assert "config/local.example.toml" in text

    def test_a_site_that_is_not_on_earth_is_refused_by_name(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path)
        tables["site"]["latitude_deg"] = 123.25
        text = str(self.refused(tmp_path, tables))
        assert "[site]" in text
        assert "latitude_deg" in text
        assert "123.25" not in text

    def test_no_catalog_path_is_refused(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path)
        del tables["survey"]["catalog_path"]
        text = str(self.refused(tmp_path, tables))
        assert "[survey]" in text
        assert "catalog_path" in text
        assert "seeingmon catalog build" in text

    def test_no_survey_table_at_all_is_refused_the_same_way(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path)
        del tables["survey"]
        text = str(self.refused(tmp_path, tables))
        assert "[survey]" in text
        assert "catalog_path" in text

    def test_a_catalog_file_that_does_not_exist_is_refused_without_its_path(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "vendor" / "missing-catalog.example"
        tables = working_tables(tmp_path, catalog_path=str(missing))
        error = self.refused(tmp_path, tables)
        assert "catalog file that [survey] catalog_path names does not exist" in str(error)
        assert str(missing) not in str(error)
        assert "missing-catalog" not in str(error)
        assert "seeingmon catalog build" in str(error)

    @pytest.mark.parametrize(
        ("content", "reason"),
        [(b"\0" * 4096, "bad magic"), (b"not a catalog", "shorter than the catalog header")],
    )
    def test_a_file_that_is_not_a_catalog_is_refused_with_the_reason_and_no_path(
        self, tmp_path: Path, content: bytes, reason: str
    ) -> None:
        other = tmp_path / "notes-of-the-night.example"
        other.write_bytes(content)
        error = self.refused(tmp_path, working_tables(tmp_path, catalog_path=str(other)))
        assert "[survey] catalog_path names is not a cap catalog" in str(error)
        assert reason in str(error)
        assert "notes-of-the-night" not in str(error)
        assert "seeingmon catalog build" in str(error)

    def test_a_catalog_path_that_names_a_folder_is_refused_too(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path, catalog_path=str(tmp_path))
        assert "does not exist" in str(self.refused(tmp_path, tables))

    def test_an_unknown_solver_is_refused_by_its_name(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path, solvers=["astap", "cedar"])
        text = str(self.refused(tmp_path, tables))
        assert "[survey] solvers" in text
        assert "cedar" in text

    def test_a_table_that_is_not_valid_is_refused_with_its_name_and_no_value(
        self, tmp_path: Path
    ) -> None:
        tables = working_tables(tmp_path, solver_timeout_seconds=987.5)  # not a key
        text = str(self.refused(tmp_path, tables))
        assert "[survey]" in text
        assert "solver_timeout_seconds" in text
        assert "987.5" not in text

    def test_an_alignment_table_that_is_not_valid_is_refused(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path)
        tables["alignment"] = {"aim_x_px": 5.5}  # the aim needs both of its keys
        text = str(self.refused(tmp_path, tables))
        assert "[alignment]" in text


class TestTheWarnings:
    """What the launcher can tell before the night: a solver program or folder that is not there."""

    def warnings(self, tmp_path: Path, **survey: Any) -> list[str]:
        tables = working_tables(tmp_path)
        tables["survey"].update(survey)
        return real_plan(tmp_path, tables).warnings

    def test_an_installed_solver_gets_no_warning(self, tmp_path: Path) -> None:
        assert self.warnings(tmp_path) == []

    def test_a_missing_program_gets_a_warning_that_names_the_solver_and_the_setting(
        self, tmp_path: Path
    ) -> None:
        (found,) = self.warnings(tmp_path, astap_command=NOT_THERE)
        assert "solver astap" in found
        assert "[survey] astap_command" in found
        assert "PATH" in found
        assert NOT_THERE not in found  # the value of the setting stays out of the console

    def test_a_program_on_the_path_is_found_without_a_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda name: "found" if name == "astap-x" else None)
        assert self.warnings(tmp_path, astap_command="astap-x") == []
        assert len(self.warnings(tmp_path, astap_command="astap-y")) == 1

    def test_a_command_with_arguments_is_checked_by_its_program(self, tmp_path: Path) -> None:
        command = f"{stub_program(tmp_path, 'wrapper.example')} --some-argument"
        assert self.warnings(tmp_path, astap_command=command) == []
        (found,) = self.warnings(tmp_path, astap_command=f"{NOT_THERE} --some-argument")
        assert "astap_command" in found

    def test_a_command_that_the_adapters_split_wrongly_gets_the_warning_too(
        self, tmp_path: Path
    ) -> None:
        # The adapters split a command like a POSIX shell, which drops a backslash, so a Windows
        # path needs forward slashes or quotes. The check splits the same way.
        (found,) = self.warnings(tmp_path, astap_command="folder\\astap_cli.example")
        assert "astap_command" in found
        (broken,) = self.warnings(tmp_path, astap_command='"folder/never closed')
        assert "astap_command" in broken

    def test_a_command_that_a_shell_split_gets_wrong_comes_with_a_hint(
        self, tmp_path: Path
    ) -> None:
        (backslash,) = self.warnings(tmp_path, astap_command="folder\\astap_cli.example")
        assert "forward slashes" in backslash
        assert "double quotes" in backslash
        (space,) = self.warnings(tmp_path, astap_command="folder/astap cli.example")
        assert "forward slashes" in space
        (plain,) = self.warnings(tmp_path, astap_command=NOT_THERE)
        assert "forward slashes" not in plain  # nothing suggests a problem with the split

    def test_every_listed_solver_is_checked(self, tmp_path: Path) -> None:
        found = self.warnings(
            tmp_path,
            solvers=["astrometry.net", "astap"],
            solve_field_command=NOT_THERE,
            astap_command=NOT_THERE,
            index_dir=str(tmp_path / "no-index"),
        )
        assert sum("solve_field_command" in line for line in found) == 1
        assert sum("astap_command" in line for line in found) == 1
        assert sum("index_dir" in line for line in found) == 1

    def test_astrometry_net_needs_index_files_in_its_folder(self, tmp_path: Path) -> None:
        index = tmp_path / "index"
        index.mkdir()
        options: dict[str, Any] = {
            "solvers": ["astrometry.net"],
            "solve_field_command": stub_program(tmp_path, "solve-field.example"),
            "index_dir": str(index),
        }
        (empty,) = self.warnings(tmp_path, **options)
        assert "index_dir" in empty
        assert str(index) not in empty
        (index / "index-4207.fits").write_bytes(b"")
        assert self.warnings(tmp_path, **options) == []

    def test_astrometry_net_without_an_index_folder_setting_warns(self, tmp_path: Path) -> None:
        options: dict[str, Any] = {
            "solvers": ["astrometry.net"],
            "solve_field_command": stub_program(tmp_path, "solve-field.example"),
        }
        (found,) = self.warnings(tmp_path, **options)
        assert "index_dir" in found

    def test_astap_needs_no_folder_setting_but_a_named_folder_has_to_exist(
        self, tmp_path: Path
    ) -> None:
        assert self.warnings(tmp_path, astap_database_dir="") == []  # astap finds its own
        (found,) = self.warnings(tmp_path, astap_database_dir=str(tmp_path / "no-database"))
        assert "astap_database_dir" in found
        assert "no-database" not in found
        (tmp_path / "database").mkdir()
        assert self.warnings(tmp_path, astap_database_dir=str(tmp_path / "database")) == []

    def test_an_empty_solver_list_warns_that_nothing_can_find_the_pointing(
        self, tmp_path: Path
    ) -> None:
        (found,) = self.warnings(tmp_path, solvers=[])
        assert "solvers is empty" in found
        assert "pointing" in found

    def test_a_solver_that_is_not_listed_is_not_checked(self, tmp_path: Path) -> None:
        assert self.warnings(tmp_path, solve_field_command=NOT_THERE) == []  # only astap is listed


class TestTheBanner:
    def test_it_says_what_is_real_and_names_the_solvers_that_run(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path, solvers=["astrometry.net", "astap"])
        tables["survey"]["solve_field_command"] = stub_program(tmp_path, "solve-field.example")
        (tmp_path / "index").mkdir()
        (tmp_path / "index" / "index-4207.fits").write_bytes(b"")
        tables["survey"]["index_dir"] = str(tmp_path / "index")
        lines = banner(real_plan(tmp_path, tables))
        assert lines[0] == "Seeing monitor, real sky (asi driver): real time, full sensor."
        assert any(line.startswith("Web UI: ") for line in lines)
        text = "\n".join(lines)
        assert "Real: the camera, the system clock," in text
        assert "Nothing about the sky is simulated." in text
        assert "No pointing solution is seeded." in text
        assert "in this order: astrometry.net, astap." in text
        assert "A later run starts from the newest solution in the store." in text
        assert "ASTAP has solved real frames offline only" in text
        assert "Cover the camera by hand for a dark session." in lines
        assert lines[-1] == "Press Ctrl+C to stop."
        assert not [line for line in lines if line.startswith("Warning:")]

    def test_it_says_that_the_scheduler_follows_the_real_sun_at_the_real_site(
        self, tmp_path: Path
    ) -> None:
        lines = banner(real_plan(tmp_path))
        (sun,) = [line for line in lines if "real Sun" in line]
        assert "at your site" in sun
        assert "the measured sky" in sun
        assert "The height of the Sun does not limit the search." in sun  # the default
        assert "2 search bursts in a row" in sun
        assert not [
            line for line in lines if "synthetic site" in line or "reports no stars" in line
        ]

    def test_it_says_which_settings_of_the_owner_count(self, tmp_path: Path) -> None:
        lines = banner(real_plan(tmp_path))
        (line,) = [line for line in lines if line.startswith("Of your local configuration")]
        assert "[site], [survey], [alignment], [web], and [auth]" in line
        assert "no sink, heater, SQM-LE, or power setting" in line
        assert "20 s" in line
        assert "5 frames" in line

    def test_it_prints_no_coordinate_no_path_and_no_host(self, tmp_path: Path) -> None:
        tables = working_tables(
            tmp_path,
            index_dir=str(tmp_path / "index"),
            astap_database_dir=str(tmp_path / "database"),
            astap_command=NOT_THERE,
        )
        plan = real_plan(tmp_path, tables, text=OWNER_WITHOUT_A_SITE)
        assert plan.warnings  # the lines below include the warnings
        text = "\n".join(lines_without_the_urls(plan))
        for forbidden in (*SITE_NUMBERS, str(tmp_path), tmp_path.name, "cap.example", NOT_THERE):
            assert forbidden not in text, forbidden
        for part in ("\\", ":/", "//"):
            assert part not in text, part  # no drive, no folder separator, no URL
        for host in ("192.0.2.10", "seeing.example.test", "influx", "2001:db8"):
            assert host not in text, host

    def test_the_warnings_follow_the_notes_and_come_with_the_word(self, tmp_path: Path) -> None:
        tables = working_tables(tmp_path, astap_command=NOT_THERE)
        plan = real_plan(tmp_path, tables)
        lines = banner(plan)
        warnings = [line for line in lines if line.startswith("Warning: ")]
        assert [line.removeprefix("Warning: ") for line in warnings] == plan.warnings
        assert len(warnings) == 1
        assert lines.index(warnings[0]) > lines.index(plan.notes[-1])

    def test_it_shows_no_api_token_line_when_the_owner_has_a_hash(self, tmp_path: Path) -> None:
        plan = real_plan(tmp_path, text=OWNER_WITHOUT_A_SITE, tables=None)
        assert plan.token is None
        assert not [line for line in banner(plan) if "token" in line.lower()]

    def test_a_sensor_that_the_person_chose_gets_a_note(self, tmp_path: Path) -> None:
        plan = real_plan(tmp_path, sensor="small", sensor_given=True)
        notes = [line for line in banner(plan) if "--sensor" in line]
        assert notes == ["--sensor does not apply to the real camera, which has the full sensor."]


class TestTheLogs:
    def test_the_children_log_into_the_logs_folder_of_the_data_folder(self, tmp_path: Path) -> None:
        origin = iso_to_utc_ns("2026-10-03T21:04:05Z")
        plan = real_plan(tmp_path, origin_real_ns=origin)
        folder = tmp_path / "data" / "logs" / "20261003T210405Z"
        assert plan.log_dir == folder
        assert folder.is_dir()  # the launcher made it, so a child can open its log at once
        assert {spec.name: spec.log for spec in plan.children} == {
            name: folder / f"{name}.log" for name in ("acquire", "core", "web")
        }
        assert not list(plan.directory.rglob("*.log"))  # none in the run folder

    def test_the_banner_names_the_folder_relative_to_the_data_folder(self, tmp_path: Path) -> None:
        plan = real_plan(tmp_path, origin_real_ns=iso_to_utc_ns("2026-10-03T21:04:05Z"))
        assert plan.log_folder == "logs/20261003T210405Z"
        (line,) = [line for line in banner(plan) if line.startswith("The logs of the children")]
        assert line == (
            "The logs of the children are in the folder logs/20261003T210405Z of your data folder."
        )
        assert str(tmp_path) not in line

    def test_a_second_run_gets_a_folder_of_its_own(self, tmp_path: Path) -> None:
        first = real_plan(tmp_path, origin_real_ns=iso_to_utc_ns("2026-10-03T21:04:05Z"))
        second = real_plan(
            tmp_path, name="second", origin_real_ns=iso_to_utc_ns("2026-10-04T20:00:00Z")
        )
        assert first.log_dir != second.log_dir
        assert sorted(p.name for p in (tmp_path / "data" / "logs").iterdir()) == [
            "20261003T210405Z",
            "20261004T200000Z",
        ]

    def test_a_simulated_run_keeps_its_logs_in_its_run_folder(self, tmp_path: Path) -> None:
        plan = build_plan(
            DevOptions(), directory=tmp_path / "sim", local_file=tmp_path / "none.toml", env={}
        )
        for spec in plan.children:
            assert spec.log == plan.directory / f"{spec.name}.log"


def namespace(**changes: Any) -> argparse.Namespace:
    """The arguments of `seeingmon dev --driver asi --real-sky --data-dir kept`."""
    values: dict[str, Any] = {
        "speed": 1.0,
        "port": None,
        "sensor": None,
        "seed": 1,
        "start": None,
        "keep_data": False,
        "log_level": None,
        "driver": "asi",
        "data_dir": "kept",
        "real_sky": True,
    }
    values.update(changes)
    return argparse.Namespace(**values)


class TestTheOptionsOfTheCommand:
    def args(self, **changes: Any) -> argparse.Namespace:
        return namespace(**changes)

    def test_a_real_sky_logs_at_the_level_info_unless_the_person_asks_otherwise(self) -> None:
        assert options_from_args(self.args()).log_level == "info"
        assert options_from_args(self.args(log_level="warning")).log_level == "warning"
        assert options_from_args(self.args(log_level="debug")).log_level == "debug"

    def test_the_other_runs_log_at_the_level_warning_by_default(self) -> None:
        assert options_from_args(self.args(real_sky=False)).log_level == "warning"
        assert options_from_args(self.args(real_sky=False, driver="asi")).real_sky is False

    def test_the_level_reaches_the_children(self, tmp_path: Path) -> None:
        plan = real_plan(tmp_path, log_level="info")
        for spec in plan.children:
            assert spec.argv[-2:] == ["--log-level", "info"], spec.name

    def test_a_command_line_without_the_real_camera_is_refused_before_anything_starts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["dev", "--real-sky", "--data-dir", "kept"]) == 2
        error = capsys.readouterr().err
        assert "--real-sky needs --driver asi" in error

    def test_the_help_describes_the_real_sky(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as raised:
            main(["dev", "--help"])
        assert raised.value.code == 0
        # The help wraps a long option at its hyphens, so join the pieces again.
        text = " ".join(capsys.readouterr().out.split()).replace("- ", "-")
        assert "--real-sky" in text
        assert "needs --driver asi and --data-dir" in text
        assert "SEEINGMON_SITE__*" in text
        assert "default warning, and info with --real-sky" in text


class TestTheRefusalsOfTheLauncher:
    def args(self, **changes: Any) -> argparse.Namespace:
        return namespace(**changes)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"driver": "sim"}, "--real-sky needs --driver asi"),
            ({"driver": None}, "--real-sky needs --driver asi"),
            ({"data_dir": None}, "--real-sky needs --data-dir"),
            ({"data_dir": ""}, "--real-sky needs --data-dir"),
            ({"driver": "sim", "data_dir": None}, "--real-sky needs --driver asi"),
        ],
    )
    def test_a_real_sky_without_the_real_camera_or_a_data_folder_is_refused(
        self, tmp_path: Path, changes: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(CliError, match=message) as raised:
            run_dev(self.args(**changes), env={}, directory=tmp_path / "run")
        assert raised.value.exit_code == 2
        assert not (tmp_path / "run").exists()
        assert "kept" not in str(raised.value)

    def test_the_plan_itself_refuses_a_real_sky_without_the_real_camera(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(CliError, match="--real-sky needs --driver asi"):
            build_plan(
                DevOptions(real_sky=True, data_dir=tmp_path / "data"),
                directory=tmp_path / "run",
                local_file=tmp_path / "none.toml",
                env={},
            )
        with pytest.raises(CliError, match="--real-sky needs --data-dir"):
            build_plan(
                DevOptions(real_sky=True, acquire_driver="asi"),
                directory=tmp_path / "run",
                local_file=tmp_path / "none.toml",
                env={},
            )
        assert not (tmp_path / "run").exists()

    def test_a_missing_catalog_comes_through_run_dev_as_it_is_and_leaves_no_folder(
        self, tmp_path: Path
    ) -> None:
        local = tmp_path / "owner.toml"
        local.write_text(
            toml(working_tables(tmp_path, catalog_path=str(tmp_path / "no-catalog.example"))),
            encoding="utf-8",
        )
        args = self.args(data_dir=str(tmp_path / "kept"))
        lines: list[str] = []
        with pytest.raises(
            CliError, match=r"catalog file that \[survey\] catalog_path names"
        ) as raised:
            run_dev(args, local_file=local, env={}, out=lines.append, directory=tmp_path / "run")
        assert raised.value.exit_code == 2
        assert "cannot prepare" not in str(raised.value)  # the message is the refusal itself
        assert "no-catalog" not in str(raised.value)
        assert not (tmp_path / "run").exists()  # the run folder is gone
        assert not (tmp_path / "kept").exists()  # and nothing was written to the data folder

    def test_a_site_that_is_missing_comes_through_run_dev_with_its_remedy(
        self, tmp_path: Path
    ) -> None:
        local = tmp_path / "owner.toml"
        local.write_text(toml({"survey": working_tables(tmp_path)["survey"]}), encoding="utf-8")
        with pytest.raises(CliError, match=r"\[site\] table of your local configuration") as raised:
            run_dev(
                self.args(data_dir=str(tmp_path / "kept")),
                local_file=local,
                env={},
                directory=tmp_path / "run",
            )
        assert "untracked local/config.toml" in str(raised.value)
        assert raised.value.exit_code == 2
