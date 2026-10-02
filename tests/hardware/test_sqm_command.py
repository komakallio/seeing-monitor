"""`seeingmon hardware sqm`: one read of the configured source, and a line for a person."""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest

from seeingmon.cli import build_parser, main
from seeingmon.clock import NS_PER_S
from seeingmon.hardware import sqm_influx
from seeingmon.sinks.influx import make_opener
from tests.hardware.influx_replies import MAGNITUDE, TEMPERATURE, v1_reply, v2_csv
from tests.hardware.sqm_server import FakeSqmServer
from tests.sinks.fake_influx import Drop, FakeInfluxServer, Reply, Respond

TOKEN = "example-token-value"
CODE = "example-code-value"  # the password of a version 1 server
SENTINEL = "SENTINEL-installation-value"


@pytest.fixture
def server() -> Iterator[FakeInfluxServer]:
    with FakeInfluxServer() as running:
        yield running


@pytest.fixture(autouse=True)
def _loopback_without_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command builds its own HTTP client, so keep the proxy variables of the machine out."""
    monkeypatch.setattr(
        sqm_influx, "make_opener", partial(make_opener, use_environment_proxies=False)
    )


def write_config(tmp_path: Path, text: str) -> list[str]:
    local = tmp_path / "config.toml"
    local.write_text(text, encoding="utf-8")
    return ["hardware", "sqm", "--local-config", str(local)]


def influx_text(server: FakeInfluxServer, *, version: int = 2, extra: str = "") -> str:
    head = f'[sqm]\nsource = "influx"\n[sqm.influx]\nendpoint = "{server.url}"\n'
    if version == 2:
        head += f'org = "{SENTINEL}-org"\nbucket = "{SENTINEL}-bucket"\ntoken = "{TOKEN}"\n'
    else:
        head += (
            f'version = 1\ndatabase = "{SENTINEL}-db"\nusername = "{SENTINEL}-user"\n'
            f'password = "{CODE}"\n'
        )
    return (
        head
        + f'measurement = "{SENTINEL}-measurement"\nfield = "mag"\ntemperature_field = "temp"\n'
        + extra
        + f'[sqm.influx.tags]\n{SENTINEL.replace("-", "_")} = "{SENTINEL}-tag"\n'
    )


def fresh_v2() -> Respond:
    def make(request: object) -> Reply:
        t_utc_ns = time.time_ns() - 3 * NS_PER_S
        return Reply(200, v2_csv([("mag", t_utc_ns, MAGNITUDE), ("temp", t_utc_ns, TEMPERATURE)]))

    return Respond(make)


def fresh_v1() -> Respond:
    def make(request: object) -> Reply:
        return Reply(200, v1_reply([[time.time_ns() - 3 * NS_PER_S, MAGNITUDE, TEMPERATURE]]))

    return Respond(make)


class TestTheCommand:
    def test_the_parser_lists_hardware_and_sqm(self) -> None:
        parser = build_parser()
        assert "hardware" in parser.format_help()
        args = parser.parse_args(["hardware", "sqm"])
        assert (args.command, args.hardware_command) == ("hardware", "sqm")
        assert args.local_config is None

    def test_a_missing_subcommand_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as caught:
            main(["hardware"])
        assert caught.value.code == 2
        assert "<subcommand>" in capsys.readouterr().err

    @pytest.mark.parametrize("version", [1, 2])
    def test_a_reading_prints_one_line_and_exits_0(
        self,
        tmp_path: Path,
        server: FakeInfluxServer,
        capsys: pytest.CaptureFixture[str],
        version: int,
    ) -> None:
        server.default = fresh_v1() if version == 1 else fresh_v2()
        code = main(write_config(tmp_path, influx_text(server, version=version)))
        captured = capsys.readouterr()
        assert code == 0
        assert captured.err == ""
        (line,) = captured.out.splitlines()
        assert line.startswith("magnitude 21.43 mag/arcsec^2, temperature 3.4 C, age 3.")
        assert line.endswith(" s (source influx)")

    def test_the_output_names_no_setting_of_the_installation(
        self, tmp_path: Path, server: FakeInfluxServer, capsys: pytest.CaptureFixture[str]
    ) -> None:
        server.default = fresh_v2()
        main(write_config(tmp_path, influx_text(server)))
        text = capsys.readouterr().out
        assert SENTINEL not in text
        assert server.url not in text
        assert TOKEN not in text

    @pytest.mark.parametrize("version", [1, 2])
    def test_the_reader_uses_the_settings_of_the_file(
        self, tmp_path: Path, server: FakeInfluxServer, version: int
    ) -> None:
        server.default = fresh_v1() if version == 1 else fresh_v2()
        main(write_config(tmp_path, influx_text(server, version=version)))
        (request,) = server.requests
        assert request.method == ("GET" if version == 1 else "POST")
        assert SENTINEL + "-measurement" in (request.query.get("q", [""])[0] + request.body)

    def test_a_reader_that_is_not_enabled_is_read_too(
        self, tmp_path: Path, server: FakeInfluxServer
    ) -> None:
        server.default = fresh_v2()
        text = influx_text(server).replace(
            'source = "influx"', 'enabled = false\nsource = "influx"'
        )
        assert main(write_config(tmp_path, text)) == 0

    def test_a_failed_read_exits_1_with_one_line_that_names_no_setting(
        self, tmp_path: Path, server: FakeInfluxServer, capsys: pytest.CaptureFixture[str]
    ) -> None:
        server.default = Reply(
            401, f'{{"message": "bad token for {SENTINEL}-org at {server.url}"}}'
        )
        code = main(write_config(tmp_path, influx_text(server)))
        captured = capsys.readouterr()
        assert code == 1
        assert captured.out == ""
        (line,) = captured.err.splitlines()
        assert line.startswith("seeingmon: error: InfluxDB answered HTTP 401")
        assert SENTINEL not in line
        assert server.url not in line
        assert TOKEN not in line

    @pytest.mark.parametrize(
        ("reply", "words"),
        [
            (Reply(503), "HTTP 503"),
            (Reply(200, ""), "no reading"),
            (Reply(200, "garbage"), "not annotated CSV"),
            (Drop(), "did not answer"),
        ],
    )
    def test_each_kind_of_failure_gives_one_line(
        self,
        tmp_path: Path,
        server: FakeInfluxServer,
        capsys: pytest.CaptureFixture[str],
        reply: Reply | Drop,
        words: str,
    ) -> None:
        server.default = reply
        assert main(write_config(tmp_path, influx_text(server))) == 1
        (line,) = capsys.readouterr().err.splitlines()
        assert words in line

    def test_a_stale_point_says_how_old_it_is(
        self, tmp_path: Path, server: FakeInfluxServer, capsys: pytest.CaptureFixture[str]
    ) -> None:
        old = time.time_ns() - 5000 * NS_PER_S
        server.default = Reply(200, v2_csv([("mag", old, MAGNITUDE)]))
        assert main(write_config(tmp_path, influx_text(server))) == 1
        (line,) = capsys.readouterr().err.splitlines()
        assert "newest reading is 50" in line
        assert "max_age_s is 600 s" in line

    def test_a_token_variable_that_is_not_set_is_an_error_that_names_the_variable(
        self,
        tmp_path: Path,
        server: FakeInfluxServer,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("THE_TOKEN_VARIABLE", raising=False)
        text = influx_text(server).replace(f'token = "{TOKEN}"', 'token_env = "THE_TOKEN_VARIABLE"')
        assert main(write_config(tmp_path, text)) == 1
        (line,) = capsys.readouterr().err.splitlines()
        assert "THE_TOKEN_VARIABLE" in line

    def test_a_token_variable_that_is_set_authorizes_the_request(
        self, tmp_path: Path, server: FakeInfluxServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("THE_TOKEN_VARIABLE", TOKEN)
        server.default = fresh_v2()
        text = influx_text(server).replace(f'token = "{TOKEN}"', 'token_env = "THE_TOKEN_VARIABLE"')
        assert main(write_config(tmp_path, text)) == 0
        assert server.requests[0].headers["authorization"] == f"Token {TOKEN}"

    def test_the_source_influx_without_the_table_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(write_config(tmp_path, '[sqm]\nsource = "influx"\n')) == 1
        (line,) = capsys.readouterr().err.splitlines()
        assert "[sqm.influx]" in line

    def test_a_table_with_a_wrong_key_is_an_error_that_names_the_key_and_no_value(
        self, tmp_path: Path, server: FakeInfluxServer, capsys: pytest.CaptureFixture[str]
    ) -> None:
        text = influx_text(server, extra='tokn = "oops"\n')
        assert main(write_config(tmp_path, text)) == 1
        err = capsys.readouterr().err
        assert "tokn" in err
        assert "oops" not in err
        assert TOKEN not in err
        assert SENTINEL not in err

    def test_the_tcp_source_reads_the_unit_and_prints_the_same_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = FakeSqmServer()
        try:
            text = f'[sqm]\nhost = "127.0.0.1"\nport = {fake.port}\nread_timeout_s = 1.0\n'
            code = main(write_config(tmp_path, text))
        finally:
            fake.close()
        assert code == 0
        assert capsys.readouterr().out == (
            "magnitude 21.37 mag/arcsec^2, temperature 3.5 C, age 0.0 s (source tcp)\n"
        )

    def test_the_tcp_source_without_a_host_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(write_config(tmp_path, "[sqm]\nenabled = false\n")) == 1
        assert "host is not set" in capsys.readouterr().err

    def test_a_tcp_unit_that_does_not_answer_is_an_error_without_the_address(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake = FakeSqmServer()
        fake.script.append("drop")
        try:
            text = f'[sqm]\nhost = "127.0.0.1"\nport = {fake.port}\nread_timeout_s = 1.0\n'
            code = main(write_config(tmp_path, text))
        finally:
            fake.close()
        err = capsys.readouterr().err
        assert code == 1
        assert "127.0.0.1" not in err
        assert str(fake.port) not in err
