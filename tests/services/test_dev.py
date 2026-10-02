"""The dev launcher: the settings of each child, the isolation of the run, and a whole run.

The tests use documentation addresses (RFC 5737 and RFC 3849) for the web settings of the owner. No
test binds one of them: the plan only decides what the children receive and which URLs it prints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sep", reason="the survey path needs the survey extra")
pytest.importorskip("fastapi", reason="the web settings come from the web extra")

from seeingmon.cli import CliError
from seeingmon.clock import iso_to_utc_ns
from seeingmon.config import load_config
from seeingmon.fastpath import FastPathConfig
from seeingmon.scheduler import SchedulerConfig
from seeingmon.scheduler.ephemeris import sun_elevation_deg
from seeingmon.services.config import ServicesConfig
from seeingmon.services.dev import (
    DevOptions,
    DevPlan,
    banner,
    build_plan,
    child_environment,
    default_start_utc_ns,
    flatten_env,
    owner_settings,
    render_env_value,
    run_dev,
    url_for,
)
from seeingmon.services.web.auth import hash_token, verify_token
from seeingmon.services.web.config import AuthSettings, WebSettings

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
