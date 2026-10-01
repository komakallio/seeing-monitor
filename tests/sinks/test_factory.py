"""`build_sinks` turns the `[sinks]` section into sinks, and the template shows valid sections."""

from __future__ import annotations

import random
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from seeingmon.clock import VirtualClock
from seeingmon.config import Config, ConfigError, load_config
from seeingmon.sinks.base import Sink
from seeingmon.sinks.config import SinksSection
from seeingmon.sinks.factory import build_sinks
from seeingmon.sinks.forwarder import Forwarder
from seeingmon.sinks.influx import InfluxSink, make_opener
from seeingmon.sinks.timescale import TimescaleSink
from seeingmon.store.config import ForwarderConfig
from seeingmon.store.db import Store
from tests.sinks.fake_influx import FakeInfluxServer
from tests.sinks.line_protocol import parse_line, split_lines
from tests.store.builders import NS_PER_S, T0, make_health

TOKEN = "example-token-value"
CODE = "example-code-value"


def fake_psycopg(calls: list[dict[str, Any]]) -> ModuleType:
    module = ModuleType("psycopg")

    def connect(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "connection"

    module.__dict__["connect"] = connect
    return module


def load(text: str, tmp_path: Path, env: dict[str, str] | None = None) -> Config:
    local = tmp_path / "config.toml"
    local.write_text(text, encoding="utf-8")
    return load_config(local_file=local, env=env or {})


FILE = """
[sinks.lab_influx]
kind = "influx"
endpoint = "https://influx.example.org:8086"
org = "example-org"
bucket = "seeing"
token_env = "LAB_INFLUX_TOKEN"

[sinks.archive]
kind = "timescale"
host = "timescale.example.org"
database = "seeing"
user = "example-user"
password_env = "ARCHIVE_PASSWORD"  # pragma: allowlist secret
record_types = ["seeing_window"]

[sinks.off]
kind = "influx"
enabled = false
endpoint = "https://influx.example.org:8086"
org = "o"
bucket = "b"
"""


class TestBuildSinks:
    def test_no_sinks_section_means_no_sinks(self, tmp_path: Path) -> None:
        assert build_sinks(load("", tmp_path), env={}) == []

    def test_each_table_becomes_a_sink_of_its_kind_in_file_order(self, tmp_path: Path) -> None:
        env = {"LAB_INFLUX_TOKEN": TOKEN, "ARCHIVE_PASSWORD": CODE}
        sinks = build_sinks(
            load(FILE, tmp_path), env=env, import_module=lambda name: fake_psycopg([])
        )
        assert [sink.name for sink in sinks] == ["lab_influx", "archive"]  # the third is disabled
        assert isinstance(sinks[0], InfluxSink)
        assert isinstance(sinks[1], TimescaleSink)
        assert all(isinstance(sink, Sink) for sink in sinks)
        assert sinks[0].accepts("health")
        assert sinks[1].accepts("seeing_window")
        assert not sinks[1].accepts("health")  # the record_types filter reaches the sink

    def test_a_secret_comes_from_the_named_environment_variable(self, tmp_path: Path) -> None:
        calls: list[dict[str, Any]] = []
        env = {"LAB_INFLUX_TOKEN": TOKEN, "ARCHIVE_PASSWORD": CODE}
        sinks = build_sinks(
            load(FILE, tmp_path), env=env, import_module=lambda name: fake_psycopg(calls)
        )
        timescale = sinks[1]
        assert isinstance(timescale, TimescaleSink)
        assert calls == []  # nothing connects while the sinks are built
        timescale._connect()
        assert calls[0]["password"] == CODE
        assert calls[0]["host"] == "timescale.example.org"

    def test_a_missing_variable_names_the_variable_and_the_sink_and_no_value(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            build_sinks(
                load(FILE, tmp_path),
                env={"ARCHIVE_PASSWORD": CODE},
                import_module=lambda name: fake_psycopg([]),
            )
        message = str(caught.value)
        assert "LAB_INFLUX_TOKEN" in message
        assert "lab_influx" in message
        assert CODE not in message

    def test_a_missing_driver_fails_when_the_sinks_are_built(self, tmp_path: Path) -> None:
        def missing(name: str) -> ModuleType:
            raise ModuleNotFoundError(name)

        env = {"LAB_INFLUX_TOKEN": TOKEN, "ARCHIVE_PASSWORD": CODE}
        with pytest.raises(ConfigError, match=r"timescale.*extra"):
            build_sinks(load(FILE, tmp_path), env=env, import_module=missing)

    def test_a_disabled_sink_needs_nothing(self, tmp_path: Path) -> None:
        text = FILE.replace('token_env = "LAB_INFLUX_TOKEN"', "enabled = false")
        text = text.replace('record_types = ["seeing_window"]', "enabled = false")
        assert build_sinks(load(text, tmp_path), env={}) == []

    def test_a_direct_token_in_the_file_works_without_the_environment(self, tmp_path: Path) -> None:
        text = (
            '[sinks.lab]\nkind = "influx"\nendpoint = "https://influx.example.org"\n'
            f'org = "o"\nbucket = "b"\ntoken = "{TOKEN}"\n'
        )
        (sink,) = build_sinks(load(text, tmp_path), env={})
        assert TOKEN not in repr(sink)

    def test_environment_variables_alone_can_define_a_sink(self, tmp_path: Path) -> None:
        env = {
            "SEEINGMON_SINKS__LAB__KIND": "influx",
            "SEEINGMON_SINKS__LAB__ENDPOINT": "https://influx.example.org",
            "SEEINGMON_SINKS__LAB__ORG": "o",
            "SEEINGMON_SINKS__LAB__BUCKET": "b",
            "SEEINGMON_SINKS__LAB__TOKEN": "98765",
        }
        config = load_config(local_file=tmp_path / "none.toml", env=env)
        (sink,) = build_sinks(config, env=env)
        assert sink.name == "lab"
        assert "98765" not in repr(sink)

    def test_an_invalid_section_is_a_configuration_error(self, tmp_path: Path) -> None:
        text = '[sinks.lab]\nkind = "influx"\nendpoint = "https://influx.example.org"\n'
        with pytest.raises(ConfigError, match="version 2 needs an org and a bucket"):
            build_sinks(load(text, tmp_path), env={})

    def test_the_os_environment_is_the_default_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LAB_INFLUX_TOKEN", TOKEN)
        text = FILE.split("[sinks.archive]")[0]
        (sink,) = build_sinks(load(text, tmp_path))
        assert sink.name == "lab_influx"


class TestTheModuleStaysLight:
    def test_the_factory_imports_without_a_postgres_driver(self) -> None:
        code = "import sys; sys.modules['psycopg'] = None; import seeingmon.sinks.factory"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_importing_the_package_does_not_load_a_driver_or_an_http_client(self) -> None:
        code = (
            "import sys, seeingmon.sinks; "
            "print([m for m in ('psycopg', 'urllib.request', 'sqlite3') if m in sys.modules])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.stdout.strip() == "[]"


class TestTheTemplate:
    @staticmethod
    def is_toml(line: str) -> bool:
        try:
            tomllib.loads(line)
        except tomllib.TOMLDecodeError:
            return False
        return True

    @staticmethod
    def sinks_block(repo_root: Path) -> str:
        """The sinks part of the template, from its heading to the next blank line.

        Other lanes add their own sections after it, so the tests read this block only.
        """
        text = (repo_root / "config" / "local.example.toml").read_text(encoding="utf-8")
        return text[text.index("# Result sinks.") :].split("\n\n", 1)[0]

    def uncommented(self, repo_root: Path) -> str:
        """The TOML lines of the sinks block, without their `# `. Lines of prose do not parse."""
        candidates = [
            line[2:]
            for line in self.sinks_block(repo_root).splitlines()
            if line.startswith("# [sinks.") or re.match(r"# [a-z_]+ = ", line)
        ]
        return "\n".join(line for line in candidates if self.is_toml(line))

    def test_the_commented_examples_are_valid_sections(self, repo_root: Path) -> None:
        parsed = tomllib.loads(self.uncommented(repo_root))
        sinks = Config(parsed).section("sinks", SinksSection).root
        assert set(sinks) == {"lab_influx", "old_influx", "archive"}
        assert sinks["lab_influx"].kind == "influx"
        assert sinks["old_influx"].kind == "influx"
        assert sinks["archive"].kind == "timescale"

    def test_the_examples_name_every_required_key_of_their_kind(self, repo_root: Path) -> None:
        parsed = tomllib.loads(self.uncommented(repo_root))
        assert {"kind", "endpoint", "org", "bucket"} <= set(parsed["sinks"]["lab_influx"])
        assert {"kind", "endpoint", "database", "version"} <= set(parsed["sinks"]["old_influx"])
        assert {"kind", "host", "database", "user"} <= set(parsed["sinks"]["archive"])

    def test_the_examples_build_with_placeholder_variables(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        parsed = tomllib.loads(self.uncommented(repo_root))
        env = {}
        for settings in parsed["sinks"].values():
            for key in ("token_env", "password_env"):
                if key in settings:
                    env[settings[key]] = "example-value"
        sinks = build_sinks(Config(parsed), env=env, import_module=lambda name: fake_psycopg([]))
        assert [sink.name for sink in sinks] == ["lab_influx", "old_influx", "archive"]

    def test_the_sinks_block_holds_no_real_looking_endpoint(self, repo_root: Path) -> None:
        hosts = re.findall(r"https?://([^/:\"]+)", self.sinks_block(repo_root))
        assert hosts
        for host in hosts:
            assert host.endswith(".example.org") or host.endswith(".example.com")


class TestEndToEnd:
    def test_a_station_forwards_its_results_through_the_configured_sink(
        self, tmp_path: Path
    ) -> None:
        with FakeInfluxServer() as server:
            text = (
                f'[sinks.lab]\nkind = "influx"\nendpoint = "{server.url}"\norg = "o"\n'
                'bucket = "seeing"\ntoken_env = "LAB_TOKEN"\n'
            )
            config = load(text, tmp_path)
            (sink,) = build_sinks(config, env={"LAB_TOKEN": TOKEN})
            assert isinstance(sink, InfluxSink)
            # The sink above uses the system proxy settings, so send through one without them.
            sink._opener = make_opener(use_environment_proxies=False)
            clock = VirtualClock(T0)
            with Store.open(tmp_path / "results.sqlite") as store:
                store.write_many([make_health(T0 + n * NS_PER_S) for n in range(5)])
                forwarder = Forwarder(store, [sink], clock, ForwarderConfig(), rng=random.Random(1))
                forwarder.run_once()
                assert forwarder.sink_backlog() == {"lab": 0}
        (request,) = server.requests
        assert request.headers["authorization"] == f"Token {TOKEN}"
        assert request.query["bucket"] == ["seeing"]
        assert [parse_line(line).timestamp for line in split_lines(request.body)] == [
            T0 + n * NS_PER_S for n in range(5)
        ]
