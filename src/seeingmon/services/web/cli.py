"""The `seeingmon web` commands: serve the API and the UI, print the OpenAPI document, hash a token.

    seeingmon web                      serve on the addresses of the [web] section
    seeingmon web --demo               serve synthetic data and a fake core, and print the URL
    seeingmon web openapi              print the OpenAPI document (--output FILE writes it,
                                       --check FILE compares it)
    seeingmon web hash-token           hash an API token for the [auth] section

`seeingmon web` reads the layered configuration (see `seeingmon.config`). It opens the store
read-only, so it can start before `core` has made the database. It talks to `core` through the
connection layer of `[services]`, and it sends `READY=1` and the watchdog heartbeat to systemd when
the unit runs with `Type=notify` (see `seeingmon.services.web.runner`). Stop it with SIGTERM or
Ctrl+C.

`seeingmon web --demo` needs no camera, store, `core`, or token hash. It serves 24 hours of
synthetic data and a fake `core` that streams a synthetic star field (see
`seeingmon.services.web.demo`), on the addresses of the same layered `[web]` section, and it prints
the URL as the only line on the standard output. The commands of the demo take the token `demo`.

`hash-token` never takes the token on the command line, where `ps` would show it. It reads the
token from the standard input, or from a hidden prompt on a terminal, and an empty answer (or
`--generate`) makes a new random token. The hash is the only thing that the command writes to the
standard output, so `seeingmon web hash-token > hash.txt` captures it. A token that the command
generates goes to the standard error, for you to keep in a password manager.

The module imports no web library at the top, so that `seeingmon --help` stays fast and works
without the `web` extra.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seeingmon.cli import CliError, Subparsers, add_command

if TYPE_CHECKING:
    from fastapi import FastAPI

    from seeingmon.services.web.config import WebSettings

LOG_LEVELS = ("debug", "info", "warning", "error")
MIN_TOKEN_CHARS = 20
GENERATED_TOKEN_BYTES = 32
WEB_MODULES = frozenset({"fastapi", "starlette", "uvicorn", "PIL", "websockets", "h11"})
_TOKEN_CHARS = re.compile(r"[A-Za-z0-9._~+/-]+=*")
_log = logging.getLogger(__name__)


def register(subparsers: Subparsers) -> None:
    web = add_command(
        subparsers,
        "web",
        help="Run the web process (the REST API and the UI), or use one of its tools.",
        handler=_serve,
    )
    web.add_argument(
        "--demo",
        action="store_true",
        help="serve synthetic data and a fake core, and print the URL. It needs no camera, "
        "store, or token hash. The commands take the token 'demo'.",
    )
    web.add_argument("--port", type=_port, help="listen on this port (default: [web] port)")
    web.add_argument(
        "--local-config",
        type=Path,
        help="read this file instead of local/config.toml (an absent file is ignored)",
    )
    web.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        help="the log level (default: info, and warning for --demo)",
    )
    commands = web.add_subparsers(dest="web_command", metavar="<subcommand>")
    openapi = commands.add_parser(
        "openapi",
        help="Print the OpenAPI document of the REST API.",
        description=(
            "Print the OpenAPI 3.1 document that FastAPI builds from the routes. The document "
            "does not depend on any data or configuration. docs/openapi.json holds the committed "
            "copy, and a test fails when it is stale."
        ),
    )
    openapi.add_argument("--output", type=Path, metavar="FILE", help="write the document to FILE")
    openapi.add_argument(
        "--check", type=Path, metavar="FILE", help="exit with 1 when FILE differs from the document"
    )
    openapi.set_defaults(handler=_openapi)
    hash_token = commands.add_parser(
        "hash-token",
        help="Hash an API token for the [auth] section. The hash goes to the standard output.",
        description=(
            "Make the salted scrypt hash of an API token. The command reads the token from the "
            "standard input, or from a hidden prompt on a terminal. It never takes the token as "
            "an argument. An empty answer, or --generate, makes a new random token and prints "
            "it to the standard error. The hash is the only output on the standard output."
        ),
    )
    hash_token.add_argument(
        "--generate", action="store_true", help="make a new random token instead of reading one"
    )
    hash_token.set_defaults(handler=_hash_token)


# --- The server ------------------------------------------------------------------------------


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = -1
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("a port is a number from 0 to 65535")
    return value


@dataclass
class Composed:
    """The app of `seeingmon web`, its settings, and what to release after the server stopped."""

    app: FastAPI
    settings: WebSettings
    close: Callable[[], None]
    token_set: bool


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _explain(error: ModuleNotFoundError) -> Exception:
    """A short message for a missing web package, and the error itself for any other module."""
    name = (error.name or "").split(".")[0]
    if name in WEB_MODULES:
        return CliError(
            f"the web process needs the {name} package: "
            "install it with pip install 'seeingmon[web]'"
        )
    return error


def compose(local_config: Path | None = None) -> Composed:
    """Build the app from the layered configuration.

    Raises `ConfigError`, `ProfileError`, `AuthConfigError`, or `IpcError` for a configuration that
    the process cannot run with. The store is not opened yet (see `ReopeningReader`).
    """
    from seeingmon.config import load_config
    from seeingmon.profile.summary import profile_summary
    from seeingmon.services.config import ServicesConfig
    from seeingmon.services.web.app import create_app
    from seeingmon.services.web.config import AuthSettings, WebSettings
    from seeingmon.services.web.core_client import RpcCoreClient
    from seeingmon.services.web.images import ImageStore
    from seeingmon.services.web.reader import ReopeningReader
    from seeingmon.store.layout import DataLayout

    config = load_config(local_file=local_config)
    web = config.section("web", WebSettings)
    auth = config.section("auth", AuthSettings)
    services = config.section("services", ServicesConfig)
    token_hash = auth.load_token_hash()
    clock = services.clock.build()
    layout = DataLayout.from_config(config)
    reader = ReopeningReader(layout.db_path, clock)
    core = RpcCoreClient.from_config(services, web.core, clock=clock)
    app = create_app(
        web,
        reader,
        ImageStore(layout, web.images),
        core,
        clock=clock,
        token_hash=token_hash,
        profile=profile_summary(config.profile),
        config=config.effective(redact=True, omit_site=True),
        station_id=config.station_id,
    )

    def close() -> None:
        core.close()
        reader.close()

    return Composed(app, web, close, token_hash is not None)


def _stop_on_signals(request_stop: Callable[[str], None]) -> None:
    """Route SIGINT, SIGTERM, and SIGBREAK to `request_stop`. The call works in the main thread."""
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum: int, frame: object) -> None:
        request_stop(f"signal {signum}")

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, name, None)  # SIGBREAK exists on Windows only
        if number is not None:
            signal.signal(number, handler)


def _serve_demo(args: argparse.Namespace) -> int:
    """Serve the demo: synthetic data, a fake core, and the layered `[web]` settings."""
    _configure_logging(args.log_level or "warning")
    try:
        from seeingmon.config import ConfigError, load_config
        from seeingmon.profile import ProfileError
        from seeingmon.profile.summary import profile_summary
        from seeingmon.services.web.config import WebSettings
        from seeingmon.services.web.demo import build_demo
        from seeingmon.services.web.runner import BindError, WebRunner

        try:
            config = load_config(local_file=args.local_config)
            web = config.section("web", WebSettings)
            profile = profile_summary(config.profile)
            view = config.effective(redact=True, omit_site=True)
        except (ConfigError, ProfileError) as error:
            raise CliError(str(error)) from None
        demo = build_demo(web, profile=profile, config=view)
        runners: list[WebRunner] = []

        def print_url() -> None:
            print(runners[0].url, flush=True)

        runner = WebRunner(demo.app, web, port=args.port, on_started=print_url)
        runners.append(runner)
        _stop_on_signals(runner.request_stop)
        try:
            return runner.run()
        except BindError as error:
            raise CliError(str(error)) from None
        finally:
            demo.close()
    except ModuleNotFoundError as error:
        raise _explain(error) from None


def _serve(args: argparse.Namespace) -> int:
    if args.demo:
        return _serve_demo(args)
    _configure_logging(args.log_level or "info")
    try:
        from seeingmon.config import ConfigError
        from seeingmon.profile import ProfileError
        from seeingmon.services.ipc.errors import IpcError
        from seeingmon.services.web.auth import AuthConfigError
        from seeingmon.services.web.runner import BindError, WebRunner

        try:
            composed = compose(args.local_config)
        except (ConfigError, ProfileError, AuthConfigError, IpcError) as error:
            raise CliError(str(error)) from None
        if not composed.token_set:
            _log.warning("no token hash is configured, so the API refuses every command")
        runner = WebRunner(composed.app, composed.settings, port=args.port)
        _stop_on_signals(runner.request_stop)
        try:
            return runner.run()
        except BindError as error:
            raise CliError(str(error)) from None
        finally:
            composed.close()
    except ModuleNotFoundError as error:
        raise _explain(error) from None


# --- openapi ---------------------------------------------------------------------------------


def _openapi(args: argparse.Namespace) -> int:
    try:
        from seeingmon.services.web.openapi import COMMAND, render_openapi

        text = render_openapi()
    except ModuleNotFoundError as error:
        raise _explain(error) from None
    if args.check is not None:
        try:
            current = args.check.read_text(encoding="utf-8")
        except OSError as error:
            raise CliError(f"cannot read {args.check}: {error.strerror or 'error'}") from None
        if current != text:
            raise CliError(f"{args.check} is stale: run `{COMMAND}`")
        return 0
    if args.output is not None:
        try:
            with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except OSError as error:
            raise CliError(f"cannot write {args.output}: {error.strerror or 'error'}") from None
        return 0
    sys.stdout.write(text)
    return 0


# --- hash-token ------------------------------------------------------------------------------


def _check_token(token: str) -> None:
    """Refuse a token that a client could not send, or that is too short. Never shows the token."""
    from seeingmon.services.web.auth import MAX_TOKEN_CHARS

    if not MIN_TOKEN_CHARS <= len(token) <= MAX_TOKEN_CHARS:
        raise CliError(f"a token has {MIN_TOKEN_CHARS} to {MAX_TOKEN_CHARS} characters", 2)
    if _TOKEN_CHARS.fullmatch(token) is None:
        raise CliError(
            "a token holds letters, digits, and the characters . _ ~ + / - only, "
            "because a client sends it in an HTTP header",
            2,
        )


def _read_token() -> str | None:
    """The token from a hidden prompt (on a terminal) or from the first line of the input.

    Returns `None` when the person asks for a new token with an empty answer on a terminal.
    """
    if sys.stdin.isatty():
        import getpass

        answer = getpass.getpass("Token (press Enter to make a new one): ", stream=sys.stderr)
        return answer.strip() or None
    line = sys.stdin.readline().strip()
    if not line:
        raise CliError(
            "no token on the standard input: pipe one in, or use --generate to make a new one", 2
        )
    return line


def _hash_token(args: argparse.Namespace) -> int:
    import secrets

    try:
        from seeingmon.services.web.auth import AuthConfigError, hash_token
    except ModuleNotFoundError as error:
        raise _explain(error) from None
    token = None if args.generate else _read_token()
    generated = token is None
    if token is None:
        token = secrets.token_urlsafe(GENERATED_TOKEN_BYTES)
    else:
        _check_token(token)
    try:
        hashed = hash_token(token)
    except AuthConfigError as error:
        raise CliError(str(error), 2) from None
    if generated:
        print(
            "A new token. Keep it in a password manager: the server stores only the hash.\n"
            f"token: {token}\n"
            "The hash goes to the standard output. Put it in the [auth] section of the local "
            "configuration, or in a file for the token hash credential.",
            file=sys.stderr,
        )
    print(hashed)
    return 0
