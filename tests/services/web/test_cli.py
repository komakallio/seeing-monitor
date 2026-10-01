"""The `seeingmon web` commands: serve, openapi, and hash-token."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from seeingmon.cli import main
from seeingmon.services.web import runner as runner_module
from seeingmon.services.web.auth import (
    MAX_TOKEN_CHARS,
    ScryptParams,
    bearer_token,
    hash_token,
    parse_token_hash,
    verify_token,
)
from seeingmon.services.web.openapi import COMMAND, render_openapi
from seeingmon.services.web.runner import WebRunner, bind_sockets
from tests.services.web.helpers import TOKEN
from tests.services.web.server import API, LOOPBACK, fetch, wait_started

KEY = "a-test-key-with-32-characters-long"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("SEEINGMON_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    """Run `seeingmon <argv>` and return the exit code, the standard output, and the error."""
    capsys.readouterr()
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def feed_stdin(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    def feed(text: str) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(text))

    return feed


# --- The command tree ------------------------------------------------------------------------


def test_the_web_command_lists_its_options_and_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["web", "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    for word in ("--port", "--local-config", "--log-level", "openapi", "hash-token"):
        assert word in output


def test_the_top_level_help_lists_web(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "web" in capsys.readouterr().out


@pytest.mark.parametrize("port", ["-1", "65536", "x", "8080.5"])
def test_a_bad_port_is_a_usage_error(port: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["web", "--port", port])
    assert raised.value.code == 2
    assert "0 to 65535" in capsys.readouterr().err


# --- hash-token ------------------------------------------------------------------------------


def test_the_hash_of_a_token_from_the_input_goes_to_the_output_alone(
    capsys: pytest.CaptureFixture[str], feed_stdin: Callable[[str], None]
) -> None:
    feed_stdin(f"{TOKEN}\n")
    code, out, err = run_cli(capsys, "web", "hash-token")
    assert code == 0
    assert err == ""
    lines = out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("$scrypt$")
    assert TOKEN not in out
    assert verify_token(TOKEN, lines[0]) is True
    assert verify_token(TOKEN + "x", lines[0]) is False
    assert parse_token_hash(lines[0]).params == ScryptParams()


def test_each_run_draws_a_new_salt(
    capsys: pytest.CaptureFixture[str], feed_stdin: Callable[[str], None]
) -> None:
    hashes = set()
    for _ in range(2):
        feed_stdin(f"{TOKEN}\n")
        hashes.add(run_cli(capsys, "web", "hash-token")[1])
    assert len(hashes) == 2


def test_generate_makes_a_token_for_the_error_stream_and_a_hash_for_the_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_cli(capsys, "web", "hash-token", "--generate")
    assert code == 0
    (hashed,) = out.splitlines()
    token_lines = [line for line in err.splitlines() if line.startswith("token: ")]
    assert len(token_lines) == 1
    token = token_lines[0].removeprefix("token: ")
    assert len(token) >= 40
    assert bearer_token(f"Bearer {token}") == token  # a client can send it
    assert verify_token(token, hashed) is True
    assert token not in out
    assert "password manager" in err


def test_two_generated_tokens_differ(capsys: pytest.CaptureFixture[str]) -> None:
    first = run_cli(capsys, "web", "hash-token", "--generate")[2]
    second = run_cli(capsys, "web", "hash-token", "--generate")[2]
    assert first != second


def test_an_empty_answer_at_the_prompt_makes_a_new_token(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import getpass

    prompts: list[str] = []

    def answer(prompt: str = "", stream: object = None) -> str:
        prompts.append(prompt)
        return ""

    monkeypatch.setattr(sys, "stdin", type("Tty", (io.StringIO,), {"isatty": lambda self: True})())
    monkeypatch.setattr(getpass, "getpass", answer)
    code, out, err = run_cli(capsys, "web", "hash-token")
    assert code == 0
    assert out.startswith("$scrypt$")
    assert "token: " in err
    assert len(prompts) == 1
    assert "Enter" in prompts[0]


def test_a_token_typed_at_the_prompt_is_hashed_and_never_echoed(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import getpass

    monkeypatch.setattr(sys, "stdin", type("Tty", (io.StringIO,), {"isatty": lambda self: True})())
    monkeypatch.setattr(getpass, "getpass", lambda prompt="", stream=None: f" {TOKEN} ")
    code, out, err = run_cli(capsys, "web", "hash-token")
    assert code == 0
    assert verify_token(TOKEN, out.strip()) is True
    assert TOKEN not in out + err


def test_the_token_is_never_an_argument(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["web", "hash-token", TOKEN])
    assert raised.value.code == 2
    assert TOKEN not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "no token on the standard input"),
        ("\n", "no token on the standard input"),
        ("short-token\n", "20 to 256 characters"),
        ("x" * (MAX_TOKEN_CHARS + 1) + "\n", "20 to 256 characters"),
        ("a token with spaces in it, long enough\n", "letters, digits"),
        ("a-token-with-a-colon:and-more-text\n", "letters, digits"),
        ("a-token-with-unicode-ä-long-enough\n", "letters, digits"),
    ],
)
def test_a_token_that_cannot_work_is_refused_without_echoing_it(
    text: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
    feed_stdin: Callable[[str], None],
) -> None:
    feed_stdin(text)
    code, out, err = run_cli(capsys, "web", "hash-token")
    assert code == 2
    assert out == ""
    assert message in err
    assert text.strip() == "" or text.strip() not in err


# --- openapi ---------------------------------------------------------------------------------


def test_openapi_prints_the_document(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run_cli(capsys, "web", "openapi")
    assert code == 0
    assert err == ""
    assert out == render_openapi()
    assert json.loads(out)["info"]["title"] == "Seeing monitor API"


def test_openapi_writes_a_file_with_unix_line_ends(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "openapi.json"
    code, out, _err = run_cli(capsys, "web", "openapi", "--output", str(target))
    assert code == 0
    assert out == ""
    data = target.read_bytes()
    assert b"\r\n" not in data
    assert data.decode("utf-8") == render_openapi()


def test_openapi_check_passes_for_a_current_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "openapi.json"
    run_cli(capsys, "web", "openapi", "--output", str(target))
    assert run_cli(capsys, "web", "openapi", "--check", str(target))[0] == 0


def test_openapi_check_fails_for_a_stale_file_and_names_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "openapi.json"
    target.write_text("{}\n", encoding="utf-8")
    code, _out, err = run_cli(capsys, "web", "openapi", "--check", str(target))
    assert code == 1
    assert "stale" in err
    assert COMMAND in err


def test_openapi_check_fails_for_a_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = run_cli(capsys, "web", "openapi", "--check", str(tmp_path / "none.json"))
    assert code == 1
    assert "cannot read" in err


def test_openapi_reports_a_file_that_cannot_be_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _out, err = run_cli(capsys, "web", "openapi", "--output", str(tmp_path / "no" / "x.json"))
    assert code == 1
    assert "cannot write" in err


# --- serve -----------------------------------------------------------------------------------


@pytest.fixture
def recorded_runners(monkeypatch: pytest.MonkeyPatch) -> list[WebRunner]:
    """Make `seeingmon web` build a runner that the test can reach, to stop it."""
    runners: list[WebRunner] = []

    class Recording(WebRunner):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            runners.append(self)

    monkeypatch.setattr(runner_module, "WebRunner", Recording)
    return runners


def write_config(path: Path, tmp_path: Path, *, token_hash: str | None, web: str = "") -> Path:
    lines = [
        'station_id = "test-station"',
        "[paths]",
        f"data_dir = {json.dumps(str(tmp_path / 'data'))}",
        "[services]",
        f'connection_key = "{KEY}"',
        "[web]",
        "port = 0",
        "shutdown_timeout_s = 1.0",
        web,
    ]
    if token_hash is not None:
        lines += ["[auth]", f'token_hash = "{token_hash}"']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class Serving:
    """`seeingmon web` running in a thread, with a handle on its runner."""

    def __init__(self, runners: list[WebRunner], argv: list[str]) -> None:
        self.runners = runners
        self.code: int | None = None
        self.thread = threading.Thread(target=self._run, args=(argv,), daemon=True)

    def _run(self, argv: list[str]) -> None:
        self.code = main(argv)

    def __enter__(self) -> Serving:
        self.thread.start()
        for _ in range(2000):
            if self.runners:
                break
            assert self.thread.is_alive(), f"the command ended at start with code {self.code}"
            time.sleep(0.01)
        wait_started(self.runners[0], self.thread, lambda: self.code)
        return self

    @property
    def port(self) -> int:
        assert self.runners[0].port is not None
        return self.runners[0].port

    def __exit__(self, *exc_info: object) -> None:
        self.runners[0].request_stop("test")
        self.thread.join(20)
        assert not self.thread.is_alive()


@pytest.fixture
def token_hash() -> str:
    return hash_token(TOKEN, params=ScryptParams(ln=10, r=8, p=1))


@pytest.fixture
def serving(
    tmp_path: Path, recorded_runners: list[WebRunner], token_hash: str
) -> Iterator[Serving]:
    config = write_config(tmp_path / "local.toml", tmp_path, token_hash=token_hash)
    with Serving(recorded_runners, ["web", "--local-config", str(config)]) as running:
        yield running


def test_the_command_serves_the_configured_api_and_exits_with_zero(
    serving: Serving,
) -> None:
    port = serving.port
    status, _headers, body = fetch(LOOPBACK, port, f"{API}/status")
    assert status == 200
    document = json.loads(body)
    assert document["station_id"] == "test-station"
    assert document["core"]["reachable"] is False  # no core runs
    assert document["ui"]["commands_enabled"] is True
    page, _headers, html = fetch(LOOPBACK, port, "/")
    assert page == 200
    assert b"<html" in html.lower()
    serving.runners[0].request_stop("test")
    serving.thread.join(20)
    assert serving.code == 0


def test_the_store_may_be_missing_and_health_says_so(serving: Serving) -> None:
    status, _headers, body = fetch(LOOPBACK, serving.port, f"{API}/health")
    assert status == 503
    assert "store_unreadable" in json.loads(body)["reasons"]


def test_the_profile_and_the_configuration_are_served_without_private_values(
    serving: Serving, tmp_path: Path
) -> None:
    status, _headers, body = fetch(LOOPBACK, serving.port, f"{API}/profile")
    assert status == 200
    assert json.loads(body)["id"]
    status, _headers, body = fetch(LOOPBACK, serving.port, f"{API}/config")
    assert status == 200
    text = body.decode("utf-8")
    for private in (KEY, str(tmp_path), "scrypt"):
        assert private not in text
    assert "<redacted>" in text


def test_a_command_needs_the_token_that_the_hash_belongs_to(serving: Serving) -> None:
    port = serving.port
    burst = f"{API}/commands/burst"
    body = b"{}"
    json_type = {"Content-Type": "application/json"}
    refused = fetch(LOOPBACK, port, burst, method="POST", headers=json_type, body=body)
    assert refused[0] == 401
    wrong = {**json_type, "Authorization": "Bearer not-the-token-not-the-token"}
    assert fetch(LOOPBACK, port, burst, method="POST", headers=wrong, body=body)[0] == 401
    right = {**json_type, "Authorization": f"Bearer {TOKEN}"}
    accepted = fetch(LOOPBACK, port, burst, method="POST", headers=right, body=body)
    assert accepted[0] == 503  # the token passed, and core does not answer
    assert json.loads(accepted[2])["error"]["code"] == "core_unavailable"


def test_without_a_token_hash_the_command_warns_and_refuses_commands(
    tmp_path: Path,
    recorded_runners: list[WebRunner],
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = write_config(tmp_path / "local.toml", tmp_path, token_hash=None)
    argv = ["web", "--local-config", str(config)]
    with caplog.at_level(logging.WARNING), Serving(recorded_runners, argv) as running:
        status, _headers, body = fetch(
            LOOPBACK,
            running.port,
            f"{API}/commands/burst",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=b"{}",
        )
        assert status == 403
        assert json.loads(body)["error"]["code"] == "commands_disabled"
    assert any("no token hash" in record.getMessage() for record in caplog.records)


def test_the_port_option_overrides_the_configuration(
    tmp_path: Path, recorded_runners: list[WebRunner], token_hash: str
) -> None:
    (blocker,) = bind_sockets([LOOPBACK], 0)
    taken = int(blocker.getsockname()[1])
    blocker.close()
    config = write_config(tmp_path / "local.toml", tmp_path, token_hash=token_hash)
    argv = ["web", "--local-config", str(config), "--port", str(taken)]
    with Serving(recorded_runners, argv) as running:
        assert running.port == taken
        assert fetch(LOOPBACK, taken, f"{API}/status")[0] == 200


def test_an_allowed_host_of_the_configuration_reaches_the_server(
    tmp_path: Path, recorded_runners: list[WebRunner], token_hash: str
) -> None:
    config = write_config(
        tmp_path / "local.toml",
        tmp_path,
        token_hash=token_hash,
        web='allowed_hosts = ["pi.example"]\nextra_bind_addresses = ["127.0.0.1"]',
    )
    with Serving(recorded_runners, ["web", "--local-config", str(config)]) as running:
        assert running.runners[0].addresses == (LOOPBACK,)
        assert fetch(LOOPBACK, running.port, f"{API}/status", host="pi.example")[0] == 200
        assert fetch(LOOPBACK, running.port, f"{API}/status", host="other.example")[0] == 400


def test_a_taken_port_ends_the_command_with_a_message_that_names_the_address(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], token_hash: str
) -> None:
    (blocker,) = bind_sockets([LOOPBACK], 0)
    try:
        port = int(blocker.getsockname()[1])
        config = write_config(tmp_path / "local.toml", tmp_path, token_hash=token_hash)
        code, _out, err = run_cli(capsys, "web", "--local-config", str(config), "--port", str(port))
    finally:
        blocker.close()
    assert code == 1
    assert err.startswith("seeingmon: error: cannot listen on ")
    assert f"{LOOPBACK}:{port}" in err
    assert "Traceback" not in err


def test_a_wildcard_bind_address_is_refused_with_the_key_in_the_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], token_hash: str
) -> None:
    config = write_config(
        tmp_path / "local.toml", tmp_path, token_hash=token_hash, web='bind_address = "0.0.0.0"'
    )
    code, _out, err = run_cli(capsys, "web", "--local-config", str(config))
    assert code == 1
    assert "bind_address" in err
    assert "0.0.0.0" not in err.replace("not to 0.0.0.0 or ::", "")
    assert "Traceback" not in err


def test_a_wildcard_in_allowed_hosts_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], token_hash: str
) -> None:
    config = write_config(
        tmp_path / "local.toml",
        tmp_path,
        token_hash=token_hash,
        web='allowed_hosts = ["*.private-name.example"]',
    )
    code, _out, err = run_cli(capsys, "web", "--local-config", str(config))
    assert code == 1
    assert "allowed_hosts" in err
    assert "private-name" not in err


def test_a_missing_connection_key_says_where_to_set_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "local.toml"
    config.write_text(
        f"[paths]\ndata_dir = {json.dumps(str(tmp_path / 'data'))}\n", encoding="utf-8"
    )
    code, _out, err = run_cli(capsys, "web", "--local-config", str(config))
    assert code == 1
    assert "SEEINGMON_SERVICES__CONNECTION_KEY" in err


def test_a_missing_data_directory_setting_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "local.toml"
    config.write_text(f'[services]\nconnection_key = "{KEY}"\n', encoding="utf-8")
    code, _out, err = run_cli(capsys, "web", "--local-config", str(config))
    assert code == 1
    assert "data_dir" in err


def test_a_malformed_token_hash_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "local.toml", tmp_path, token_hash="not-a-hash")
    code, _out, err = run_cli(capsys, "web", "--local-config", str(config))
    assert code == 1
    assert "hash-token" in err


def test_the_module_imports_no_web_library_at_the_top() -> None:
    """`seeingmon --help` must stay fast and work without the `web` extra."""
    import subprocess

    probe = (
        "import sys; import seeingmon.services.web.cli; "
        "bad = [m for m in ('fastapi', 'uvicorn', 'starlette', 'PIL') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
