"""The dev launcher: the settings of each child, the isolation of the run, and a whole run.

The tests use documentation addresses (RFC 5737 and RFC 3849) for the web settings of the owner. No
test binds one of them: the plan only decides what the children receive and which URLs it prints.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("fastapi", reason="the web settings come from the web extra")

from seeingmon.cli import CliError, build_parser
from seeingmon.clock import iso_to_utc_ns
from seeingmon.config import load_config
from seeingmon.fastpath import FastPathConfig
from seeingmon.scheduler import SchedulerConfig
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.services import dev
from seeingmon.services.config import ServicesConfig
from seeingmon.services.dev import (
    ES_CONTINUOUS,
    ES_SYSTEM_REQUIRED,
    KEEP_AWAKE_NOTE,
    DevOptions,
    DevPlan,
    KeepAwake,
    banner,
    build_plan,
    child_environment,
    default_start_utc_ns,
    flatten_env,
    keeps_awake,
    options_from_args,
    owner_settings,
    render_env_value,
    run_dev,
    url_for,
)
from seeingmon.services.web.auth import hash_token, verify_token
from seeingmon.services.web.config import AuthSettings, WebSettings
from seeingmon.survey.config import SurveyConfig

OWNER_TOKEN_HASH = hash_token(
    "an-owner-token-of-twenty-or-more-characters"
)  # pragma: allowlist secret
OWNER = f"""
station_id = "owner-station"
profile = "owner-profile"
[paths]
data_dir = "owner-data-dir"
[services]
connection_key = "owner-connection-key-aaaaaaaaaaaaaaaaaaaaaaaa"  # pragma: allowlist secret
core_address = "owner-core-address"
[site]
latitude_deg = 12.5
longitude_deg = 34.5
[sinks.influx]
url = "https://influx.example.test:8086"
token = "owner-sink-token-bbbbbbbbbbbbbbbbbbbb"  # pragma: allowlist secret
[web]
bind_address = "192.0.2.10"
extra_bind_addresses = ["2001:db8::10", "198.51.100.7"]
port = 8123
allowed_hosts = ["seeing.example.test"]
[auth]
token_hash = "{OWNER_TOKEN_HASH}"
"""
SINK_TOKEN = "owner-sink-token-bbbbbbbbbbbbbbbbbbbb"  # pragma: allowlist secret
FORBIDDEN = (
    "owner-station",
    "owner-profile",
    "owner-data-dir",
    "owner-connection-key",
    "owner-core-address",
    "influx.example.test",
    "owner-sink-token",
    "12.5",
    "34.5",
)


def plan_for(
    tmp_path: Path,
    local_text: str = "",
    env: dict[str, str] | None = None,
    name: str = "run",
    **options: Any,
) -> DevPlan:
    local = tmp_path / f"{name}-owner.toml"
    local.write_text(local_text, encoding="utf-8")
    return build_plan(
        DevOptions(**options),
        directory=tmp_path / name,
        local_file=local,
        env=env if env is not None else {},
    )


def child(plan: DevPlan, name: str) -> Any:
    return next(spec for spec in plan.children if spec.name == name)


def seeingmon_env(spec: Any) -> dict[str, str]:
    return {k: v for k, v in spec.env.items() if k.startswith("SEEINGMON_")}


@pytest.fixture(scope="module")
def owner_plan(tmp_path_factory: pytest.TempPathFactory) -> DevPlan:
    return plan_for(tmp_path_factory.mktemp("owner"), OWNER)


class TestTheWebSettingsOfTheOwner:
    def test_the_web_child_gets_the_web_and_auth_sections_in_its_environment(
        self, owner_plan: DevPlan
    ) -> None:
        env = child(owner_plan, "web").env
        assert env["SEEINGMON_WEB__BIND_ADDRESS"] == '"192.0.2.10"'
        assert env["SEEINGMON_WEB__EXTRA_BIND_ADDRESSES"] == '["2001:db8::10", "198.51.100.7"]'
        assert env["SEEINGMON_WEB__PORT"] == "8123"
        assert env["SEEINGMON_WEB__ALLOWED_HOSTS"] == '["seeing.example.test"]'
        assert json.loads(env["SEEINGMON_AUTH__TOKEN_HASH"]) == OWNER_TOKEN_HASH

    def test_the_web_child_reads_them_back_as_valid_settings(
        self, owner_plan: DevPlan, tmp_path: Path
    ) -> None:
        spec = child(owner_plan, "web")
        config = load_config(local_file=tmp_path / "absent.toml", env=seeingmon_env(spec))
        web = config.section("web", WebSettings)
        assert web.bind_address == "192.0.2.10"
        assert web.extra_bind_addresses == ("2001:db8::10", "198.51.100.7")
        assert web.port == 8123
        assert web.allowed_hosts == ("seeing.example.test",)
        auth = config.section("auth", AuthSettings)
        assert auth.token_hash is not None
        assert auth.token_hash.get_secret_value() == OWNER_TOKEN_HASH

    def test_only_the_web_child_sees_them(self, owner_plan: DevPlan) -> None:
        for name in ("acquire", "core"):
            names = child(owner_plan, name).env
            assert not [k for k in names if k.startswith(("SEEINGMON_WEB__", "SEEINGMON_AUTH__"))]

    def test_every_bind_address_gets_a_url_and_an_ipv6_address_goes_in_brackets(
        self, owner_plan: DevPlan
    ) -> None:
        assert owner_plan.urls() == [
            "http://192.0.2.10:8123/",
            "http://[2001:db8::10]:8123/",
            "http://198.51.100.7:8123/",
        ]
        lines = banner(owner_plan)
        assert [line for line in lines if line.startswith("Web UI:")] == [
            "Web UI: http://192.0.2.10:8123/",
            "Web UI: http://[2001:db8::10]:8123/",
            "Web UI: http://198.51.100.7:8123/",
        ]
        assert not any("token" in line.lower() for line in lines)  # the owner has a hash

    def test_a_wildcard_prints_the_loopback_url_and_the_addresses_of_the_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dev, "device_addresses", lambda: ("198.51.100.2", "192.0.2.9"))
        plan = plan_for(tmp_path, '[web]\nbind_address = "0.0.0.0"\nport = 8123\n')
        assert plan.bind_addresses == ["0.0.0.0"]
        assert plan.urls() == [
            "http://127.0.0.1:8123/",
            "http://198.51.100.2:8123/",
            "http://192.0.2.9:8123/",
        ]
        web_lines = [line for line in banner(plan) if line.startswith("Web UI:")]
        assert web_lines == [f"Web UI: {url}" for url in plan.urls()]

    def test_the_ipv6_wildcard_prints_the_ipv6_loopback_url_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dev, "device_addresses", lambda: ("198.51.100.2",))
        local = '[web]\nbind_address = "0.0.0.0"\nextra_bind_addresses = ["::"]\nport = 8123\n'
        plan = plan_for(tmp_path, local)
        assert plan.urls() == [
            "http://127.0.0.1:8123/",
            "http://198.51.100.2:8123/",
            "http://[::1]:8123/",
        ]

    def test_a_wildcard_with_no_network_prints_the_loopback_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dev, "device_addresses", lambda: ())
        plan = plan_for(tmp_path, '[web]\nbind_address = "0.0.0.0"\nport = 8123\n')
        assert plan.urls() == ["http://127.0.0.1:8123/"]

    def test_the_launcher_waits_for_the_web_child_on_the_loopback_for_a_wildcard(
        self, tmp_path: Path
    ) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            plan = plan_for(tmp_path, '[web]\nbind_address = "0.0.0.0"\n')
            plan.port = int(listener.getsockname()[1])

            class Alive:
                running = True

            assert dev.wait_for_web(plan, Alive(), 5.0) is True  # type: ignore[arg-type]
        finally:
            listener.close()

    def test_the_url_of_an_ipv6_address_is_bracketed_once(self) -> None:
        assert url_for("2001:db8::1", 80) == "http://[2001:db8::1]:80/"
        assert url_for("[2001:db8::1]", 80) == "http://[2001:db8::1]:80/"
        assert url_for("192.0.2.1", 80) == "http://192.0.2.1:80/"

    def test_without_web_settings_the_ui_listens_on_the_loopback_interface_only(
        self, tmp_path: Path
    ) -> None:
        plan = plan_for(tmp_path)
        assert plan.bind_addresses == ["127.0.0.1"]
        assert plan.port == 8080
        env = child(plan, "web").env
        assert env["SEEINGMON_WEB__BIND_ADDRESS"] == '"127.0.0.1"'
        assert env["SEEINGMON_WEB__PORT"] == "8080"

    def test_each_default_applies_only_when_the_owner_sets_no_value(self, tmp_path: Path) -> None:
        only_address = plan_for(tmp_path, '[web]\nbind_address = "192.0.2.20"\n', name="a")
        assert (only_address.bind_addresses, only_address.port) == (["192.0.2.20"], 8080)
        only_port = plan_for(tmp_path, "[web]\nport = 8222\n", name="b")
        assert (only_port.bind_addresses, only_port.port) == (["127.0.0.1"], 8222)

    def test_the_option_beats_the_port_of_the_owner(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path, OWNER, port=9100)
        assert plan.port == 9100
        assert plan.urls()[0] == "http://192.0.2.10:9100/"

    def test_an_environment_variable_beats_the_file_as_in_every_layer(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path, OWNER, env={"SEEINGMON_WEB__PORT": "9001"})
        assert plan.port == 9001

    def test_the_settings_of_the_owner_are_what_the_two_sections_hold_and_nothing_else(
        self, tmp_path: Path
    ) -> None:
        local = tmp_path / "owner.toml"
        local.write_text(OWNER, encoding="utf-8")
        env = {
            "SEEINGMON_WEB__PORT": "9001",
            "SEEINGMON_SINKS__INFLUX__TOKEN": "another-owner-secret",  # pragma: allowlist secret
            "SEEINGMON_SERVICES__CONNECTION_KEY": "another-owner-key",  # pragma: allowlist secret
            "HOME": "x",
        }
        web, auth = owner_settings(local, env)
        assert set(web) == {"bind_address", "extra_bind_addresses", "port", "allowed_hosts"}
        assert web["port"] == 9001
        assert set(auth) == {"token_hash"}


class TestTheIsolationOfTheRun:
    def test_nothing_else_of_the_owner_reaches_a_child(self, owner_plan: DevPlan) -> None:
        everything = json.dumps(
            [[spec.argv, spec.env] for spec in owner_plan.children], sort_keys=True
        )
        for text in FORBIDDEN:
            assert text not in everything, text

    def test_the_run_files_hold_nothing_of_the_owner(self, owner_plan: DevPlan) -> None:
        for path in owner_plan.directory.rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                for text in FORBIDDEN:
                    assert text.encode() not in data, (path.name, text)

    def test_the_run_sets_its_own_keys_in_every_child(self, owner_plan: DevPlan) -> None:
        envs = {spec.name: seeingmon_env(spec) for spec in owner_plan.children}
        for env in envs.values():
            assert json.loads(env["SEEINGMON_STATION_ID"]) == "dev"
            assert json.loads(env["SEEINGMON_SERVICES__CONNECTION_KEY"]) == owner_plan.key
            assert json.loads(env["SEEINGMON_SERVICES__CORE_ADDRESS"]) == owner_plan.core_endpoint
            assert (
                json.loads(env["SEEINGMON_SERVICES__ACQUIRE_ADDRESS"])
                == owner_plan.acquire_endpoint
            )
            assert Path(json.loads(env["SEEINGMON_PATHS__DATA_DIR"])).parent == (
                owner_plan.directory
            )
        clocks = {
            tuple(
                env[f"SEEINGMON_SERVICES__CLOCK__{key}"]
                for key in ("START_UTC_NS", "SPEED", "ORIGIN_REAL_NS")
            )
            for env in envs.values()
        }
        assert len(clocks) == 1  # the three processes agree on the time

    def test_acquire_runs_the_simulator_and_core_gets_the_catalog_and_the_seed(
        self, owner_plan: DevPlan
    ) -> None:
        acquire = seeingmon_env(child(owner_plan, "acquire"))
        assert json.loads(acquire["SEEINGMON_SERVICES__ACQUIRE__DRIVER"]) == "sim"
        core = seeingmon_env(child(owner_plan, "core"))
        assert Path(json.loads(core["SEEINGMON_SURVEY__CATALOG_PATH"])).is_file()
        assert Path(json.loads(core["SEEINGMON_SERVICES__CORE__SEED_SOLUTION_FILE"])).is_file()

    def test_the_full_sensor_waits_longer_for_the_camera_than_the_small_one(
        self, tmp_path: Path
    ) -> None:
        small = plan_for(tmp_path, name="small")
        full = plan_for(tmp_path, name="full", sensor="full")
        margin = "SEEINGMON_SERVICES__ACQUIRE__READ_TIMEOUT_MARGIN_S"
        assert margin not in seeingmon_env(child(small, "acquire"))
        assert float(seeingmon_env(child(full, "acquire"))[margin]) == 20.0
        absent = tmp_path / "absent.toml"
        for plan, expected in ((small, 0.5), (full, 20.0)):
            core = load_config(local_file=absent, env=seeingmon_env(child(plan, "core")))
            assert core.section("scheduler", SchedulerConfig).loop.read_timeout_margin_s == expected

    def test_each_child_loads_its_environment_as_a_valid_configuration(
        self, owner_plan: DevPlan, tmp_path: Path
    ) -> None:
        for spec in owner_plan.children:
            config = load_config(local_file=tmp_path / "absent.toml", env=seeingmon_env(spec))
            services = config.section("services", ServicesConfig)
            assert services.clock.kind == "scaled"
            assert config.station_id == "dev"
            assert config.profile.id == "sim-small"
        core = load_config(
            local_file=tmp_path / "absent.toml", env=seeingmon_env(child(owner_plan, "core"))
        )
        assert core.section("scheduler", SchedulerConfig).fast.analysis_window_s == 20.0
        assert core.section("scheduler", SchedulerConfig).fast.window_s == 60.0  # a simulated run
        assert core.section("fastpath", FastPathConfig).window_s == 20.0

    def test_no_child_reads_the_local_file_of_the_owner(self, owner_plan: DevPlan) -> None:
        for spec in owner_plan.children:
            index = spec.argv.index("--local-config")
            assert not Path(spec.argv[index + 1]).exists()

    def test_the_children_start_with_the_interpreter_of_the_launcher(
        self, owner_plan: DevPlan
    ) -> None:
        for spec in owner_plan.children:
            assert spec.argv[:3] == [sys.executable, "-m", "seeingmon"]

    def test_the_environment_of_a_child_drops_the_settings_and_the_credentials(self) -> None:
        parent = {
            "PATH": "/bin",
            "SEEINGMON_SERVICES__CONNECTION_KEY": "secret",  # pragma: allowlist secret
            "CREDENTIALS_DIRECTORY": "/run/credentials/x",
            "NOTIFY_SOCKET": "/run/notify",
            "OTHER": "kept",
        }
        env = child_environment(parent)
        assert env["PATH"] == "/bin"
        assert env["OTHER"] == "kept"
        assert env["PYTHONUNBUFFERED"] == "1"
        assert not [k for k in env if k.startswith("SEEINGMON_")]
        assert "CREDENTIALS_DIRECTORY" not in env
        assert "NOTIFY_SOCKET" not in env

    def test_a_secret_never_goes_on_a_command_line(self, owner_plan: DevPlan) -> None:
        for spec in owner_plan.children:
            line = " ".join(spec.argv)
            assert owner_plan.key not in line
            assert OWNER_TOKEN_HASH not in line


class TestTheCover:
    def test_the_simulated_camera_gets_a_cover_file_in_the_run_folder(
        self, owner_plan: DevPlan, tmp_path: Path
    ) -> None:
        assert owner_plan.cover_file == owner_plan.directory / "cover"
        assert not owner_plan.cover_file.exists()  # the camera starts uncovered
        config = load_config(
            local_file=tmp_path / "absent.toml", env=seeingmon_env(child(owner_plan, "acquire"))
        )
        options = config.section("services", ServicesConfig).acquire.driver_options
        assert options["cover_file"] == str(owner_plan.cover_file)

    def test_only_acquire_hears_of_the_file(self, owner_plan: DevPlan) -> None:
        assert owner_plan.cover_file is not None
        for name in ("core", "web"):
            assert str(owner_plan.cover_file) not in json.dumps(child(owner_plan, name).env)

    def test_the_banner_says_in_one_line_how_to_cover_and_uncover_the_camera(
        self, owner_plan: DevPlan
    ) -> None:
        assert owner_plan.cover_file is not None
        lines = [line for line in banner(owner_plan) if "cover" in line]
        assert len(lines) == 1
        assert str(owner_plan.cover_file) in lines[0]
        assert "create the file" in lines[0]
        assert "uncover the camera, delete the file" in lines[0]

    def test_another_driver_has_nothing_to_cover(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path, acquire_driver="replay")
        assert plan.cover_file is None
        assert not [line for line in banner(plan) if "cover" in line]
        assert "COVER_FILE" not in json.dumps(child(plan, "acquire").env)

    def test_a_test_can_name_another_file(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path, extra_sim={"cover_file": str(tmp_path / "elsewhere")})
        acquire = seeingmon_env(child(plan, "acquire"))
        key = "SEEINGMON_SERVICES__ACQUIRE__DRIVER_OPTIONS__COVER_FILE"
        assert json.loads(acquire[key]) == str(tmp_path / "elsewhere")

    def test_the_dark_session_of_a_dev_run_is_short(
        self, owner_plan: DevPlan, tmp_path: Path
    ) -> None:
        core = load_config(
            local_file=tmp_path / "absent.toml", env=seeingmon_env(child(owner_plan, "core"))
        )
        dark = core.section("survey", SurveyConfig).dark
        assert (dark.frames, dark.bias_frames, dark.poll_s) == (5, 5, 2.0)
        assert dark.exposure_s == 30.0  # the exposure of the survey stays

    def test_a_test_can_set_the_dark_session_again(self, tmp_path: Path) -> None:
        plan = plan_for(
            tmp_path, core_overrides={"survey": {"dark": {"frames": 3, "wait_timeout_s": 60.0}}}
        )
        core = load_config(
            local_file=tmp_path / "absent.toml", env=seeingmon_env(child(plan, "core"))
        )
        dark = core.section("survey", SurveyConfig).dark
        assert (dark.frames, dark.bias_frames, dark.wait_timeout_s) == (3, 5, 60.0)


LIBRARY_VARIABLE = "SEEINGMON_ASI__LIBRARY_PATH"


class TestTheRealCamera:
    """`--driver asi`: the plan of a run on the real camera. No test starts a real camera."""

    def real(self, tmp_path: Path, local_text: str = "", **options: Any) -> DevPlan:
        return plan_for(tmp_path, local_text, acquire_driver="asi", **options)

    def configuration(self, plan: DevPlan, name: str, tmp_path: Path) -> Any:
        return load_config(
            local_file=tmp_path / "absent.toml", env=seeingmon_env(child(plan, name))
        )

    def test_every_child_runs_the_real_profile_on_the_system_clock(self, tmp_path: Path) -> None:
        plan = self.real(tmp_path, sensor="small")  # the sensor does not matter
        assert plan.real
        for name in ("acquire", "core", "web"):
            config = self.configuration(plan, name, tmp_path)
            assert config.profile.id == "asi294mm-gs250"  # the full sensor
            assert config.section("services", ServicesConfig).clock.kind == "system"
        assert plan.start_utc_ns == plan.origin_real_ns  # the plan starts now
        assert abs(plan.origin_real_ns - time.time_ns()) < 120 * 10**9

    def test_acquire_runs_the_asi_driver_with_no_simulator_option(self, tmp_path: Path) -> None:
        plan = self.real(tmp_path)
        services = self.configuration(plan, "acquire", tmp_path).section("services", ServicesConfig)
        assert services.acquire.driver == "asi"
        assert services.acquire.driver_options == {}
        assert services.acquire.gap_factor == ServicesConfig().acquire.gap_factor  # not 1000
        assert plan.cover_file is None
        assert not [k for k in child(plan, "acquire").env if "DRIVER_OPTIONS" in k]

    def test_the_fast_stream_keeps_the_exposure_of_the_profile(self, tmp_path: Path) -> None:
        plan = self.real(tmp_path)
        core = self.configuration(plan, "core", tmp_path)
        scheduler = core.section("scheduler", SchedulerConfig)
        assert scheduler.fast.exposure_us == 2000  # 2 ms, and not the 50 ms of the simulator
        assert scheduler.loop.read_timeout_margin_s == 0.5
        assert scheduler.fast.analysis_window_s == 20.0  # the windows stay short for the UI
        simulated = self.configuration(plan_for(tmp_path, name="sim"), "core", tmp_path)
        assert simulated.section("scheduler", SchedulerConfig).fast.exposure_us == 50_000

    def test_the_sky_and_the_site_stay_synthetic(self, tmp_path: Path) -> None:
        plan = self.real(tmp_path)
        core = child(plan, "core").env
        assert Path(json.loads(core["SEEINGMON_SURVEY__CATALOG_PATH"])).is_file()
        assert Path(json.loads(core["SEEINGMON_SERVICES__CORE__SEED_SOLUTION_FILE"])).is_file()
        assert float(core["SEEINGMON_SITE__LATITUDE_DEG"]) == 55.0

    def test_the_banner_names_the_camera_and_says_what_the_sky_reports(
        self, tmp_path: Path
    ) -> None:
        lines = banner(self.real(tmp_path))
        assert lines[0] == "Seeing monitor, real camera (asi driver): real time, full sensor."
        assert any(line.startswith("Web UI: ") for line in lines)
        assert sum("reports no stars" in line for line in lines) == 1
        assert sum("stays in safe while the Sun is above -3 degrees" in line for line in lines) == 1
        assert "Cover the camera by hand for a dark session." in lines
        assert not [line for line in lines if "simulated camera" in line]  # no cover file
        assert not [line for line in lines if "--sensor" in line]  # nobody chose a sensor

    def test_a_sensor_that_the_person_chose_gets_a_one_line_note(self, tmp_path: Path) -> None:
        plan = self.real(tmp_path, sensor="small", sensor_given=True)
        notes = [line for line in banner(plan) if "--sensor" in line]
        assert notes == ["--sensor does not apply to the real camera, which has the full sensor."]

    def test_the_simulator_banner_is_what_it_was(self, tmp_path: Path) -> None:
        lines = banner(plan_for(tmp_path))
        assert lines[0].startswith("Seeing monitor, simulated sky: 1x speed, small sensor.")
        assert not [line for line in lines if "real camera" in line or "no stars" in line]


class TestThePriorityOfTheCaptureThread:
    """The real camera raises it, so that the reads stay on time. The simulator never does."""

    def acquire_settings(self, plan: DevPlan, tmp_path: Path) -> Any:
        config = load_config(
            local_file=tmp_path / "absent.toml", env=seeingmon_env(child(plan, "acquire"))
        )
        return config.section("services", ServicesConfig).acquire

    def test_the_real_camera_raises_the_priority_by_default(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path, acquire_driver="asi")
        assert self.acquire_settings(plan, tmp_path).raise_priority is True
        assert not [line for line in banner(plan) if "--no-raise-priority" in line]

    def test_a_comparison_run_keeps_the_normal_priority_and_says_so_once(
        self, tmp_path: Path
    ) -> None:
        plan = plan_for(tmp_path, acquire_driver="asi", raise_priority=False)
        assert self.acquire_settings(plan, tmp_path).raise_priority is False
        notes = [line for line in banner(plan) if "--no-raise-priority" in line]
        assert len(notes) == 1
        assert "normal priority" in notes[0]
        assert "default resolution" in notes[0]  # the timer of Windows

    def test_the_simulator_never_raises_it(self, tmp_path: Path) -> None:
        plans = (
            plan_for(tmp_path, name="default"),
            plan_for(tmp_path, name="asked", raise_priority=True),
        )
        for plan in plans:
            assert self.acquire_settings(plan, tmp_path).raise_priority is False
            assert not [line for line in banner(plan) if "--no-raise-priority" in line]

    def test_the_real_sky_run_follows_the_same_rule(self, tmp_path: Path) -> None:
        # The notes of a real-sky run come from the same helper as the notes of the real camera.
        from seeingmon.services.dev import _priority_notes

        assert _priority_notes(DevOptions(acquire_driver="asi")) == []
        assert len(_priority_notes(DevOptions(acquire_driver="asi", raise_priority=False))) == 1

    def test_the_command_line_option_turns_it_off(self) -> None:
        parser = build_parser()
        arguments = ["dev", "--driver", "asi"]
        on = options_from_args(parser.parse_args(arguments))
        off = options_from_args(parser.parse_args([*arguments, "--no-raise-priority"]))
        assert on.raise_priority is True
        assert off.raise_priority is False

    def test_a_namespace_without_the_option_keeps_the_default(self) -> None:
        namespace = argparse.Namespace(
            speed=1.0, port=None, sensor=None, seed=1, start=None, keep_data=False
        )
        assert options_from_args(namespace).raise_priority is True


class StandInKernel32:
    """The function of `kernel32` that the power request uses, with an answer to choose.

    Every call goes to `events`, a list that a test can share with other stand-ins.
    """

    def __init__(
        self,
        result: int = ES_CONTINUOUS,
        *,
        raises: Exception | None = None,
        raises_on_release: bool = False,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.raises = raises
        self.raises_on_release = raises_on_release
        self.calls: list[int] = []
        self.events = [] if events is None else events

    def SetThreadExecutionState(self, flags: int) -> int:  # noqa: N802 - a Windows function
        self.calls.append(flags)
        self.events.append("release" if flags == ES_CONTINUOUS else "request")
        if self.raises is not None or (self.raises_on_release and flags == ES_CONTINUOUS):
            raise self.raises or OSError("gone")
        return self.result


def holds(awake: KeepAwake) -> bool:
    """Whether the request is held. A function, so that a type checker reads it again."""
    return awake.held


class TestKeepAwake:
    """The power request that stops idle sleep. No test makes a real call."""

    def test_the_flags_are_the_ones_that_windows_documents(self) -> None:
        assert ES_CONTINUOUS == 0x80000000
        assert ES_SYSTEM_REQUIRED == 0x00000001

    def test_the_request_holds_the_system_state_until_the_release_clears_it(self) -> None:
        kernel32 = StandInKernel32()
        awake = KeepAwake(kernel32=kernel32)
        assert awake.request() is None
        assert holds(awake)
        assert kernel32.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]  # the display is not in it
        awake.release()
        assert not holds(awake)
        assert kernel32.calls[-1] == ES_CONTINUOUS  # the state is cleared, so the idle timer runs
        awake.release()
        assert len(kernel32.calls) == 2  # a second release makes no call

    def test_a_release_without_a_request_calls_nothing(self) -> None:
        kernel32 = StandInKernel32()
        KeepAwake(kernel32=kernel32).release()
        assert kernel32.calls == []

    def test_a_request_that_windows_refuses_is_a_warning_and_holds_nothing(self) -> None:
        kernel32 = StandInKernel32(result=0)  # the function returns 0 for a failure
        awake = KeepAwake(kernel32=kernel32)
        warning = awake.request()
        assert warning is not None
        assert "refused" in warning
        assert "power settings" in warning  # the person can do it by hand
        assert not holds(awake)
        awake.release()
        assert kernel32.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]  # no release of nothing

    def test_a_call_that_raises_is_a_warning_and_never_an_exception(self) -> None:
        awake = KeepAwake(kernel32=StandInKernel32(raises=OSError("no")))
        warning = awake.request()
        assert warning is not None
        assert "OSError" in warning
        assert not holds(awake)

    def test_a_release_that_raises_is_ignored(self) -> None:
        awake = KeepAwake(kernel32=StandInKernel32(raises_on_release=True))
        assert awake.request() is None
        awake.release()  # the thread is ending anyway, and Windows clears the state with it
        assert not holds(awake)

    def test_a_library_that_does_not_load_is_a_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken() -> Any:
            raise OSError("no library")

        monkeypatch.setattr(dev, "_load_kernel32", broken)
        awake = KeepAwake()
        warning = awake.request()
        assert warning is not None
        assert "OSError" in warning
        assert not holds(awake)
        awake.release()

    @pytest.mark.skipif(sys.platform != "win32", reason="the Windows library")
    def test_the_real_library_loads_privately_with_typed_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kernel32 = MagicMock()
        kernel32.SetThreadExecutionState.return_value = ES_CONTINUOUS
        opened: list[str] = []

        def win_dll(name: str) -> MagicMock:
            opened.append(name)
            return kernel32

        monkeypatch.setattr(ctypes, "WinDLL", win_dll, raising=False)
        awake = KeepAwake()
        assert awake.request() is None
        assert opened == ["kernel32"]  # a private copy, and not the shared `windll`
        # The flags do not fit a signed 32-bit integer, which is what an untyped call would take.
        assert kernel32.SetThreadExecutionState.argtypes == [ctypes.c_uint32]
        assert kernel32.SetThreadExecutionState.restype is ctypes.c_uint32
        kernel32.SetThreadExecutionState.assert_called_once_with(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        awake.release()
        kernel32.SetThreadExecutionState.assert_called_with(ES_CONTINUOUS)

    @pytest.mark.skipif(sys.platform == "win32", reason="another platform has no kernel32")
    def test_another_platform_gets_a_warning_and_never_an_import_error(self) -> None:
        warning = KeepAwake().request()
        assert warning is not None
        assert "OSError" in warning


class TestWhenTheRunHoldsTheRequest:
    @pytest.mark.parametrize(
        ("options", "system", "expected"),
        [
            (DevOptions(acquire_driver="asi"), "win32", True),
            (DevOptions(acquire_driver="asi", keep_awake=False), "win32", False),
            (DevOptions(acquire_driver="asi"), "linux", False),
            (DevOptions(acquire_driver="asi"), "darwin", False),
            (DevOptions(), "win32", False),  # the simulator is not a camera that Windows can lose
        ],
    )
    def test_only_the_real_camera_on_windows_holds_it_unless_you_opt_out(
        self, options: DevOptions, system: str, expected: bool
    ) -> None:
        assert keeps_awake(options, system) is expected

    def test_the_command_line_option_turns_it_off(self) -> None:
        parser = build_parser()
        arguments = ["dev", "--driver", "asi"]
        assert options_from_args(parser.parse_args(arguments)).keep_awake is True
        off = options_from_args(parser.parse_args([*arguments, "--no-keep-awake"]))
        assert off.keep_awake is False

    def test_a_namespace_without_the_option_keeps_the_default(self) -> None:
        namespace = argparse.Namespace(
            speed=1.0, port=None, sensor=None, seed=1, start=None, keep_data=False
        )
        assert options_from_args(namespace).keep_awake is True


class FakeChild:
    """A child that starts nothing, so that a run reaches its end at once."""

    events: ClassVar[list[str]] = []
    fail_to_start: ClassVar[str | None] = None

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self.running = True
        self.returncode = None

    def start(self) -> None:
        self.events.append(f"start {self.spec.name}")
        if self.fail_to_start == self.spec.name:
            raise CliError(f"{self.spec.name} exited with code 1")

    def stop(self, timeout_s: float = 0.0) -> None:
        self.events.append(f"stop {self.spec.name}")
        self.running = False

    def log_tail(self, lines: int = 15) -> str:
        return ""


class TestTheRunHoldsTheRequest:
    """`run_dev` asks before the children start, and it gives the request back at the end."""

    @pytest.fixture
    def events(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        shared: list[str] = []
        monkeypatch.setattr(FakeChild, "events", shared)
        monkeypatch.setattr(FakeChild, "fail_to_start", None)
        monkeypatch.setattr(dev, "Child", FakeChild)
        monkeypatch.setattr(dev, "wait_until_ready", lambda *args, **kwargs: None)
        monkeypatch.setattr(dev, "wait_for_web", lambda *args, **kwargs: True)
        return shared

    def args(self, **changes: Any) -> argparse.Namespace:
        values: dict[str, Any] = {
            "speed": 1.0,
            "port": None,
            "sensor": None,
            "seed": 1,
            "start": None,
            "keep_data": False,
            "log_level": "warning",
            "driver": "asi",
        }
        values.update(changes)
        return argparse.Namespace(**values)

    def run(
        self,
        tmp_path: Path,
        events: list[str],
        kernel32: StandInKernel32 | None,
        *,
        wait: Any = None,
        system: str = "win32",
        **changes: Any,
    ) -> tuple[list[str], int]:
        lines: list[str] = []
        code = run_dev(
            self.args(**changes),
            local_file=tmp_path / "none.toml",
            env={},
            out=lines.append,
            wait=wait or (lambda plan, children: events.append("wait")),
            directory=tmp_path / "run",
            kernel32=kernel32,
            system=system,
        )
        return lines, code

    def test_the_request_comes_before_the_children_and_the_release_before_they_stop(
        self, tmp_path: Path, events: list[str]
    ) -> None:
        kernel32 = StandInKernel32(events=events)
        _, code = self.run(tmp_path, events, kernel32)
        assert code == 0
        assert events == [
            "request",
            "start acquire",
            "start core",
            "start web",
            "wait",
            "release",
            "stop web",
            "stop core",
            "stop acquire",
        ]
        assert kernel32.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]

    def test_the_banner_says_in_one_line_what_it_does_and_what_it_does_not(
        self, tmp_path: Path, events: list[str]
    ) -> None:
        lines, _ = self.run(tmp_path, events, StandInKernel32(events=events))
        notes = [line for line in lines if "stays awake" in line]
        assert notes == [KEEP_AWAKE_NOTE]
        assert "display may still turn off" in notes[0]
        assert "Closing the lid" in notes[0]
        assert "Do nothing" in notes[0]
        assert "--no-keep-awake" in notes[0]
        assert not any(line.startswith("Warning:") for line in lines)

    def test_a_run_that_fails_to_start_gives_the_request_back(
        self, tmp_path: Path, events: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(FakeChild, "fail_to_start", "core")
        kernel32 = StandInKernel32(events=events)
        with pytest.raises(CliError, match="core exited"):
            self.run(tmp_path, events, kernel32)
        assert kernel32.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]
        assert not (tmp_path / "run").exists()

    def test_ctrl_c_gives_the_request_back(self, tmp_path: Path, events: list[str]) -> None:
        def interrupt(plan: DevPlan, children: Any) -> None:
            raise KeyboardInterrupt

        kernel32 = StandInKernel32(events=events)
        _, code = self.run(tmp_path, events, kernel32, wait=interrupt)
        assert code == 0
        assert kernel32.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]

    def test_an_error_in_the_run_gives_the_request_back(
        self, tmp_path: Path, events: list[str]
    ) -> None:
        def fail(plan: DevPlan, children: Any) -> None:
            raise RuntimeError("an unexpected error")

        kernel32 = StandInKernel32(events=events)
        with pytest.raises(RuntimeError, match="unexpected"):
            self.run(tmp_path, events, kernel32, wait=fail)
        assert kernel32.calls[-1] == ES_CONTINUOUS

    def test_a_refusal_is_a_warning_in_the_banner_and_the_run_goes_on(
        self, tmp_path: Path, events: list[str]
    ) -> None:
        kernel32 = StandInKernel32(result=0, events=events)
        lines, code = self.run(tmp_path, events, kernel32)
        assert code == 0
        assert "wait" in events  # the run went on
        warnings = [line for line in lines if line.startswith("Warning:")]
        assert len(warnings) == 1
        assert "refused the request to stay awake" in warnings[0]
        assert not any("stays awake" in line for line in lines)
        assert kernel32.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]  # nothing to give back

    def test_a_call_that_raises_is_a_warning_and_the_run_goes_on(
        self, tmp_path: Path, events: list[str]
    ) -> None:
        kernel32 = StandInKernel32(raises=OSError("no"), events=events)
        lines, code = self.run(tmp_path, events, kernel32)
        assert code == 0
        assert "wait" in events
        assert [line for line in lines if line.startswith("Warning:") and "OSError" in line]

    @pytest.mark.parametrize(
        ("changes", "system"),
        [
            ({"keep_awake": False}, "win32"),  # --no-keep-awake
            ({}, "linux"),
            ({"driver": "sim"}, "win32"),
        ],
    )
    def test_nothing_is_requested_when_the_run_does_not_need_it(
        self, tmp_path: Path, events: list[str], changes: dict[str, Any], system: str
    ) -> None:
        kernel32 = StandInKernel32(events=events)
        lines, _ = self.run(tmp_path, events, kernel32, system=system, **changes)
        assert kernel32.calls == []
        assert not any("stays awake" in line or line.startswith("Warning:") for line in lines)


class TestTheVendorLibrary:
    """The one setting of the real camera that comes from the person."""

    @pytest.fixture
    def library(self, tmp_path: Path) -> str:
        return str(tmp_path / "vendor" / "asi-library.example")

    def test_the_option_reaches_acquire_alone_and_in_its_environment(
        self, tmp_path: Path, library: str
    ) -> None:
        plan = plan_for(tmp_path, acquire_driver="asi", asi_library=library)
        assert child(plan, "acquire").env[LIBRARY_VARIABLE] == library
        for name in ("core", "web"):
            assert LIBRARY_VARIABLE not in child(plan, name).env
            assert library not in json.dumps(child(plan, name).env)
        for spec in plan.children:
            assert library not in " ".join(spec.argv)  # never on a command line
        assert library not in "\n".join(banner(plan))  # and not in the console

    def test_no_file_of_the_run_holds_it(self, tmp_path: Path, library: str) -> None:
        plan = plan_for(tmp_path, acquire_driver="asi", asi_library=library)
        for path in plan.directory.rglob("*"):
            if path.is_file():
                assert library.encode() not in path.read_bytes(), path.name

    def test_without_the_option_the_variable_of_the_launcher_is_read(
        self, tmp_path: Path, library: str
    ) -> None:
        plan = plan_for(
            tmp_path, acquire_driver="asi", env={LIBRARY_VARIABLE: library}, name="from-env"
        )
        assert child(plan, "acquire").env[LIBRARY_VARIABLE] == library
        other = str(tmp_path / "other" / "library.example")
        plan = plan_for(
            tmp_path,
            acquire_driver="asi",
            asi_library=other,
            env={LIBRARY_VARIABLE: library},
            name="beats-env",
        )
        assert child(plan, "acquire").env[LIBRARY_VARIABLE] == other  # the option wins

    def test_a_home_folder_in_the_path_is_expanded_for_acquire(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path, acquire_driver="asi", asi_library="~/vendor/library.example")
        value = child(plan, "acquire").env[LIBRARY_VARIABLE]
        assert value == str(Path("~/vendor/library.example").expanduser())
        assert "~" not in value

    def test_without_either_acquire_gets_no_variable(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path, acquire_driver="asi")
        assert LIBRARY_VARIABLE not in child(plan, "acquire").env

    def test_a_simulated_run_never_passes_it_on(self, tmp_path: Path, library: str) -> None:
        plan = plan_for(tmp_path, env={LIBRARY_VARIABLE: library})
        for spec in plan.children:
            assert LIBRARY_VARIABLE not in spec.env  # the children start clean

    def test_nothing_else_of_the_owner_reaches_a_child_of_the_real_camera(
        self, tmp_path: Path, library: str
    ) -> None:
        plan = plan_for(tmp_path, OWNER, acquire_driver="asi", asi_library=library)
        everything = json.dumps([[spec.argv, spec.env] for spec in plan.children], sort_keys=True)
        for text in FORBIDDEN:
            assert text not in everything, text
        for path in plan.directory.rglob("*"):
            if path.is_file():
                for text in FORBIDDEN:
                    assert text.encode() not in path.read_bytes(), (path.name, text)
        assert json.loads(child(plan, "web").env["SEEINGMON_WEB__PORT"]) == 8123  # the web part


class TestTheDataFolder:
    def test_without_the_option_the_run_keeps_its_own_folder(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path)
        for spec in plan.children:
            assert Path(json.loads(spec.env["SEEINGMON_PATHS__DATA_DIR"])).parent == plan.directory

    def test_the_option_names_a_folder_for_every_child(self, tmp_path: Path) -> None:
        kept = tmp_path / "kept"
        for driver in ("sim", "asi"):
            plan = plan_for(tmp_path, name=driver, acquire_driver=driver, data_dir=kept)
            for spec in plan.children:
                assert Path(json.loads(spec.env["SEEINGMON_PATHS__DATA_DIR"])) == kept


class TestTheRealCameraRefusals:
    def args(self, **changes: Any) -> argparse.Namespace:
        values: dict[str, Any] = {
            "speed": 1.0,
            "port": None,
            "sensor": None,
            "seed": 1,
            "start": None,
            "keep_data": False,
            "log_level": "warning",
            "driver": "asi",
        }
        values.update(changes)
        return argparse.Namespace(**values)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"speed": 2.0}, "--speed must be 1 with --driver asi"),
            ({"start": "2026-01-01T19:00:00Z"}, "--start does not apply with --driver asi"),
            ({"driver": "sim", "asi_library": "library"}, "--asi-library needs --driver asi"),
        ],
    )
    def test_what_the_real_camera_cannot_do_is_refused_before_anything_starts(
        self, tmp_path: Path, changes: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(CliError, match=message) as raised:
            run_dev(self.args(**changes), env={}, directory=tmp_path / "run")
        assert raised.value.exit_code == 2
        assert not (tmp_path / "run").exists()

    def test_a_library_that_does_not_exist_is_refused_without_its_path(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "vendor" / "missing-library.example"
        with pytest.raises(CliError, match="ASI library file does not exist") as raised:
            run_dev(self.args(asi_library=str(missing)), env={}, directory=tmp_path / "run")
        assert str(missing) not in str(raised.value)
        assert LIBRARY_VARIABLE in str(raised.value)  # the message names the setting
        assert not (tmp_path / "run").exists()

    def test_the_variable_of_the_launcher_gets_the_same_check(self, tmp_path: Path) -> None:
        env = {LIBRARY_VARIABLE: str(tmp_path / "missing-library.example")}
        with pytest.raises(CliError, match="ASI library file does not exist"):
            run_dev(self.args(), env=env, directory=tmp_path / "run")


class TestTheEndpoints:
    def test_every_run_gets_fresh_addresses_with_a_random_token(self, tmp_path: Path) -> None:
        first = plan_for(tmp_path, name="one")
        second = plan_for(tmp_path, name="two")
        for plan in (first, second):
            for address in (plan.core_endpoint, plan.acquire_endpoint):
                assert re.search(r"[0-9a-f]{12}", address)
        assert first.core_endpoint != second.core_endpoint
        assert first.acquire_endpoint != second.acquire_endpoint
        assert first.core_endpoint != first.acquire_endpoint

    @pytest.mark.skipif(os.name != "nt", reason="named pipes belong to Windows")
    def test_on_windows_the_address_is_a_pipe_that_the_default_names_cannot_meet(
        self, tmp_path: Path
    ) -> None:
        plan = plan_for(tmp_path)
        assert plan.core_endpoint.startswith("\\\\.\\pipe\\seeingmon-dev-")
        assert plan.core_endpoint != "\\\\.\\pipe\\seeingmon-core"

    @pytest.mark.skipif(os.name == "nt", reason="Unix sockets belong to the other platforms")
    def test_elsewhere_the_address_is_a_socket_in_the_run_folder(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path)
        assert Path(plan.core_endpoint).parent == plan.directory
        assert len(plan.core_endpoint.encode()) <= 107  # the limit of a socket path


class TestTheApiToken:
    def test_without_a_hash_the_run_makes_a_token_and_gives_the_hash_to_the_web_child(
        self, tmp_path: Path
    ) -> None:
        plan = plan_for(tmp_path)
        assert plan.token is not None
        stored = json.loads(child(plan, "web").env["SEEINGMON_AUTH__TOKEN_HASH"])
        assert verify_token(plan.token, stored)
        assert plan.token not in json.dumps([[s.argv, s.env] for s in plan.children])

    def test_the_token_is_printed_once_and_written_nowhere(self, tmp_path: Path) -> None:
        plan = plan_for(tmp_path)
        assert plan.token is not None
        lines = banner(plan)
        assert sum(plan.token in line for line in lines) == 1
        for path in plan.directory.rglob("*"):
            if path.is_file():
                assert plan.token.encode() not in path.read_bytes(), path.name

    def test_each_run_makes_another_token(self, tmp_path: Path) -> None:
        assert plan_for(tmp_path, name="a").token != plan_for(tmp_path, name="b").token

    def test_an_owner_hash_in_a_file_is_used_as_it_is_and_no_token_is_made(
        self, tmp_path: Path
    ) -> None:
        plan = plan_for(tmp_path, '[auth]\ntoken_hash_file = "hash.txt"\n')
        assert plan.token is None
        assert "SEEINGMON_AUTH__TOKEN_HASH" not in child(plan, "web").env
        assert json.loads(child(plan, "web").env["SEEINGMON_AUTH__TOKEN_HASH_FILE"]) == "hash.txt"


class TestTheSettingsFormat:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (True, "true"),
            (False, "false"),
            (3, "3"),
            (2.5, "2.5"),
            ("a b", '"a b"'),
            ([1, "x"], '[1, "x"]'),
            (("a", "b"), '["a", "b"]'),
        ],
    )
    def test_a_value_becomes_the_text_of_an_environment_variable(
        self, value: object, text: str
    ) -> None:
        assert render_env_value(value) == text

    def test_a_value_of_another_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="cannot put"):
            render_env_value(object())

    def test_nested_tables_become_double_underscore_names(self) -> None:
        flat = flatten_env({"a": {"b": 1, "c": {"d": "x"}}, "e": True}, "SEEINGMON_ROOT")
        assert flat == {
            "SEEINGMON_ROOT__A__B": "1",
            "SEEINGMON_ROOT__A__C__D": '"x"',
            "SEEINGMON_ROOT__E": "true",
        }


class TestTheStart:
    def test_the_default_start_is_in_the_dark_of_the_evening(self) -> None:
        start = default_start_utc_ns()
        assert iso_to_utc_ns("2026-01-01T12:00:00Z") < start < iso_to_utc_ns("2026-01-02T00:00:00Z")
        assert sun_elevation_deg(start, 55.0, 0.0) < -17.0

    def test_a_speed_that_is_not_positive_is_refused_before_anything_starts(self) -> None:
        args = argparse.Namespace(
            speed=0.0,
            port=None,
            sensor="small",
            seed=1,
            start=None,
            keep_data=False,
            log_level="warning",
        )
        with pytest.raises(CliError, match="--speed must be positive"):
            run_dev(args)


STUB_WEB = """
import json, os, signal, socket, sys, time
variables = {k: v for k, v in os.environ.items() if k.startswith("SEEINGMON_")}
with open(os.environ["DEV_TEST_DUMP"], "w") as handle:
    json.dump(variables, handle)
server = socket.socket()
server.bind(("127.0.0.1", int(os.environ["DEV_TEST_PORT"])))
server.listen()
signal.signal(getattr(signal, "SIGBREAK", signal.SIGTERM), lambda *a: sys.exit(0))
while True:
    time.sleep(0.2)
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def args_for(port: int) -> argparse.Namespace:
    return argparse.Namespace(
        speed=1.0,
        port=port,
        sensor="small",
        seed=1,
        start=None,
        keep_data=False,
        log_level="warning",
    )


def clean_environment(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SEEINGMON_")}
    env.update(extra)
    return env


class TestAWholeRun:
    def test_the_launcher_runs_the_whole_system_and_stops_every_child_on_ctrl_c(
        self, tmp_path: Path
    ) -> None:
        port = free_port()
        dump = tmp_path / "web-environment.json"
        local = tmp_path / "owner.toml"
        local.write_text(
            f'[web]\nbind_address = "127.0.0.1"\nextra_bind_addresses = ["::1"]\nport = {port}\n'
            f'[sinks.influx]\ntoken = "{SINK_TOKEN}"\n',
            encoding="utf-8",
        )
        lines: list[str] = []
        seen: dict[str, Any] = {}

        def stop_after_a_look(plan: DevPlan, children: Any) -> None:
            seen["children"] = dict(children)
            seen["plan"] = plan
            seen["running"] = {name: c.running for name, c in children.items()}
            raise KeyboardInterrupt  # the person at the keyboard presses Ctrl+C

        code = run_dev(
            args_for(port),
            local_file=local,
            env=clean_environment(DEV_TEST_DUMP=str(dump), DEV_TEST_PORT=str(port)),
            out=lines.append,
            web_command=[sys.executable, "-c", STUB_WEB],
            wait=stop_after_a_look,
            directory=tmp_path / "run",
        )
        assert code == 0
        assert seen["running"] == {"acquire": True, "core": True, "web": True}
        assert not any(c.running for c in seen["children"].values())  # every child stopped
        assert not (tmp_path / "run").exists()  # the temporary folder is gone
        urls = [line for line in lines if line.startswith("Web UI:")]
        assert urls == [f"Web UI: http://127.0.0.1:{port}/", f"Web UI: http://[::1]:{port}/"]
        assert sum("API token" in line for line in lines) == 1
        assert lines[-1] == "Stopping ..."
        # The web child received the owner's web section, and nothing of the sink.
        received = json.loads(dump.read_text(encoding="utf-8"))
        assert json.loads(received["SEEINGMON_WEB__PORT"]) == port
        assert "SEEINGMON_AUTH__TOKEN_HASH" in received
        assert "owner-sink-token" not in json.dumps(received)

    def test_the_data_folder_of_the_person_outlives_the_run(self, tmp_path: Path) -> None:
        port = free_port()
        kept = tmp_path / "kept"
        lines: list[str] = []

        def stop_after_a_look(plan: DevPlan, children: Any) -> None:
            raise KeyboardInterrupt

        args = args_for(port)
        args.data_dir = str(kept)
        code = run_dev(
            args,
            local_file=tmp_path / "none.toml",
            env=clean_environment(
                DEV_TEST_DUMP=str(tmp_path / "dump.json"), DEV_TEST_PORT=str(port)
            ),
            out=lines.append,
            web_command=[sys.executable, "-c", STUB_WEB],
            wait=stop_after_a_look,
            directory=tmp_path / "run",
        )
        assert code == 0
        assert not (tmp_path / "run").exists()  # the run folder is gone
        assert kept.is_dir()
        assert any(kept.iterdir())  # core put its store there
        assert not any(str(kept) in line for line in lines)  # the console names no path of yours

    def test_a_child_that_dies_ends_the_run_with_its_log_and_stops_the_others(
        self, tmp_path: Path
    ) -> None:
        port = free_port()
        lines: list[str] = []
        children: dict[str, Any] = {}

        def remember(plan: DevPlan, started: Any) -> None:
            children.update(started)

        with pytest.raises(CliError, match=r"web exited with code 7"):
            run_dev(
                args_for(port),
                local_file=tmp_path / "none.toml",
                env=clean_environment(),
                out=lines.append,
                web_command=[
                    sys.executable,
                    "-c",
                    "import sys; print('the log line'); sys.exit(7)",
                ],
                wait=remember,
                directory=tmp_path / "run",
            )
        assert not (tmp_path / "run").exists()
        assert lines[-1] == "Stopping ..."
