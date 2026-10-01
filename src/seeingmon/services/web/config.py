"""The `[web]` and `[auth]` configuration sections of the web process.

The defaults live in `config/default.d/web.toml`, and every key has the same default in the
models below. Read the sections through the configuration:

    config = load_config()
    web = config.section("web", WebSettings)
    auth = config.section("auth", AuthSettings)
    token_hash = auth.load_token_hash()

Override a key in `local/config.toml` under `[web]`, or with an environment variable such as
`SEEINGMON_WEB__BIND_ADDRESS`. The addresses and host names of one installation, and the hash of
its API token, belong to the untracked local file, to the environment, or to a systemd credential.
They never belong to a tracked file.

**Access rule.** Reads are open on the LAN by default. Set `require_token_for_reads` to ask for
the token on every read too. Every `POST` always needs the token. The `[auth]` section holds the
hash of the token (see `seeingmon.services.web.auth` for the format), and
`seeingmon web hash-token` creates one.

**Addresses and hosts.** `bind_address` and `extra_bind_addresses` name the interfaces to listen
on. `allowed_hosts` names the hosts that a request may carry in its `Host` header, in addition to
the loopback names and the bind addresses (see `seeingmon.services.web.hosts`). The two rules are
separate: an address that the process listens on is always an allowed host, and a host can be
allowed without a socket, for a name that reaches the same interface through a VPN.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator

from seeingmon.config import ConfigError, SectionModel
from seeingmon.services.web.hosts import allowed_set, normalize_entries

MIB = 1024 * 1024
LOOPBACK_V4 = "127.0.0.1"
_MAX_TOKEN_HASH_FILE_BYTES = 4096
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")


class RateLimitSettings(SectionModel):
    """How often one client may send commands, and how often it may fail the token check.

    A client is a remote address. Every `POST` counts against `commands_per_window`, whether the
    token is right or not. A failed token check also counts against `auth_failures_per_window`,
    and a client that exceeds it gets no token check at all until its failures leave the window.
    """

    commands_per_window: int = Field(30, ge=1, le=100_000)
    window_s: float = Field(60.0, gt=0, le=86_400)
    auth_failures_per_window: int = Field(5, ge=1, le=10_000)
    auth_failure_window_s: float = Field(300.0, gt=0, le=86_400)
    max_clients: int = Field(1024, ge=16, le=1_000_000)


class PagingSettings(SectionModel):
    """The size of a page of history, and how much the server reads to build it."""

    default_limit: int = Field(500, ge=1, le=10_000)
    max_limit: int = Field(2000, ge=1, le=10_000)
    max_scan_rows: int = Field(50_000, ge=100, le=5_000_000)
    default_range_hours: float = Field(24.0, gt=0, le=24 * 366 * 20)

    @model_validator(mode="after")
    def _default_fits_the_maximum(self) -> PagingSettings:
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must not exceed max_limit")
        return self


class ImageSettings(SectionModel):
    """The limits on the image list and on the files that the API serves."""

    default_list_limit: int = Field(24, ge=1, le=1000)
    max_list_limit: int = Field(200, ge=1, le=1000)
    max_preview_bytes: int = Field(8 * MIB, ge=1024)
    max_fits_bytes: int = Field(128 * MIB, ge=1024)
    scan_days: int = Field(31, ge=1, le=3660)

    @model_validator(mode="after")
    def _default_fits_the_maximum(self) -> ImageSettings:
        if self.default_list_limit > self.max_list_limit:
            raise ValueError("default_list_limit must not exceed max_list_limit")
        return self


class LiveSettings(SectionModel):
    """The alignment live view: the preview cadence and the number of clients.

    `max_fps` caps how often the server sends a frame to one client. The server always sends the
    newest frame and skips the ones in between. The server reads frames from `core` while a client
    watches, and for `idle_s` after the last one has gone.
    """

    max_fps: float = Field(2.0, gt=0, le=30)
    idle_s: float = Field(10.0, gt=0, le=3600)
    max_clients: int = Field(4, ge=1, le=64)
    stall_s: float = Field(5.0, gt=0, le=600)


class CoreLinkSettings(SectionModel):
    """How the web process talks to `core`. The address and the key come from `[services]`."""

    connect_timeout_s: float = Field(2.0, gt=0, le=60)
    rpc_timeout_s: float = Field(5.0, gt=0, le=300)
    submit_timeout_s: float = Field(10.0, gt=0, le=300)
    retry_interval_s: float = Field(1.0, ge=0, le=60)


def _interface_address(value: str, setting: str) -> str:
    """The canonical form of an address that names one interface of this device."""
    text = value.strip()
    if text.lower() == "localhost":
        return "localhost"
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        raise ValueError(
            f"set {setting} to the IP address of one interface of this device"
        ) from None
    if address.is_unspecified:
        raise ValueError(
            f"set {setting} to the address of one interface of this device, not to 0.0.0.0 or ::"
        )
    return str(address)


class WebSettings(SectionModel):
    """The `[web]` section: the addresses, the access rule, and the limits of the API.

    `bind_address` names one interface of this device, and `extra_bind_addresses` names more, such
    as the address of a VPN interface. The process listens on each of them on the same `port`, and
    never on every interface, so a wildcard address is an error. `localhost` means the IPv4
    loopback address, and `::1` is the IPv6 one.

    `allowed_hosts` lists the names and addresses that a client may put in the `Host` header (and a
    browser in the `Origin` header of a WebSocket) in addition to the loopback names and the bind
    addresses, which are always allowed. A request with another host gets an error. This rule does
    not change who may read or send commands. It keeps a page from another site, or a name that
    the owner did not list, from reaching the server through the browser of a visitor.

    `withhold_fields` lists record fields that the API serves as `null` with a `quality` note,
    because they narrow down the site (the zenith angle of Polaris equals about 90 degrees minus
    the site latitude).
    """

    bind_address: str = "127.0.0.1"
    extra_bind_addresses: tuple[str, ...] = ()
    port: int = Field(8080, ge=0, le=65535)
    allowed_hosts: tuple[str, ...] = ()

    require_token_for_reads: bool = False

    health_max_age_s: float = Field(180.0, gt=0, le=86_400)
    ui_refresh_s: float = Field(10.0, ge=1, le=3600)
    withhold_fields: tuple[str, ...] = ("zenith_angle_deg",)

    access_log: bool = False
    shutdown_timeout_s: float = Field(5.0, gt=0, le=120)
    max_body_bytes: int = Field(16 * 1024, ge=256, le=MIB)

    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    paging: PagingSettings = Field(default_factory=PagingSettings)
    images: ImageSettings = Field(default_factory=ImageSettings)
    live: LiveSettings = Field(default_factory=LiveSettings)
    core: CoreLinkSettings = Field(default_factory=CoreLinkSettings)

    @field_validator("bind_address")
    @classmethod
    def _one_interface(cls, value: str) -> str:
        return _interface_address(value, "bind_address")

    @field_validator("extra_bind_addresses")
    @classmethod
    def _more_interfaces(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(_interface_address(item, "extra_bind_addresses") for item in value)
        )

    @field_validator("allowed_hosts")
    @classmethod
    def _host_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return normalize_entries(value)

    @field_validator("withhold_fields")
    @classmethod
    def _field_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for name in value:
            if not _FIELD_NAME.fullmatch(name):
                raise ValueError("withhold_fields holds record field names in lowercase snake case")
        return value

    def listen_addresses(self) -> tuple[str, ...]:
        """The distinct addresses to listen on: `bind_address` first, then the extra addresses.

        `localhost` becomes the IPv4 loopback address, so that each entry names one socket.
        """
        names = (self.bind_address, *self.extra_bind_addresses)
        return tuple(dict.fromkeys(LOOPBACK_V4 if name == "localhost" else name for name in names))

    def allowed_host_set(self) -> frozenset[str]:
        """The hosts that a request may name. See `seeingmon.services.web.hosts.allowed_set`."""
        return allowed_set(self.bind_address, self.extra_bind_addresses, self.allowed_hosts)


class AuthSettings(SectionModel):
    """The `[auth]` section: where the hash of the API token lives.

    `token_hash` holds the hash. It stays hidden in `repr` and in the effective configuration.
    The other two sources are a systemd credential (the file `$CREDENTIALS_DIRECTORY/<name>`) and
    a file that the configuration names. `load_token_hash` reads the first source that has one.
    """

    token_hash: SecretStr | None = None
    token_hash_file: str = ""
    token_hash_credential: str = "seeingmon-token-hash"

    def load_token_hash(self, env: Mapping[str, str] | None = None) -> str | None:
        """Find the hash of the token. Returns `None` when no source has one.

        Raises `ConfigError` when a file cannot be read. The message never shows the hash.
        """
        if self.token_hash is not None and self.token_hash.get_secret_value().strip():
            return self.token_hash.get_secret_value().strip()
        environment = os.environ if env is None else env
        directory = environment.get("CREDENTIALS_DIRECTORY", "")
        if self.token_hash_credential and directory:
            candidate = Path(directory) / self.token_hash_credential
            if candidate.is_file():
                return _read_hash_file(candidate, "systemd credential")
        if self.token_hash_file.strip():
            return _read_hash_file(Path(self.token_hash_file.strip()), "token hash file")
        return None


def _read_hash_file(path: Path, description: str) -> str | None:
    try:
        if path.stat().st_size > _MAX_TOKEN_HASH_FILE_BYTES:
            raise ConfigError(f"the {description} is too large to hold a token hash")
        text = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ConfigError(
            f"cannot read the {description}: {error.strerror or type(error).__name__}"
        ) from None
    except UnicodeDecodeError:
        raise ConfigError(f"the {description} is not UTF-8 text") from None
    return text or None
