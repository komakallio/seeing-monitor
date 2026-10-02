"""The configuration of the sinks: one `[sinks.<name>]` table for each sink.

The name is the key of the table. It must be stable, because the forwarder stores the cursor of a
sink under it. Each table has a `kind` (`influx` or `timescale`) and the keys of that kind. The
configuration is local: keep every endpoint and secret in `local/config.toml` or in environment
variables, never in a tracked file. `config/local.example.toml` shows the shape.

```toml
[sinks.lab_influx]
kind = "influx"
version = 2
endpoint = "https://influx.example.org:8086"
org = "<organization>"
bucket = "<bucket>"
token_env = "LAB_INFLUX_TOKEN"   # the name of an environment variable that holds the token
```

**Secrets.** Give a secret in one of two ways. Set `token` (or `password`) in the local file, or
with the environment variable `SEEINGMON_SINKS__LAB_INFLUX__TOKEN` (the scheme of
`seeingmon.config`). Or set `token_env` (or `password_env`) to the name of any environment
variable, which suits a `systemd` `EnvironmentFile`. A secret never appears in a `repr`, in an
error message, or in the effective configuration that the `run` record stores.

**Which records.** A sink takes every table record type unless you list `record_types`. The
per-frame metrics have no table, so no sink takes them from the forwarder.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, RootModel, SecretStr, field_validator, model_validator

from seeingmon.config import ConfigError, SectionModel

SINK_NAME = re.compile(r"[a-z][a-z0-9_-]{0,31}")
SSL_MODES = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]


def check_endpoint(value: str) -> str:
    """Check the address of an InfluxDB server, and return it without a trailing slash.

    The address must use `http` or `https` and name a host, and it must hold no credentials, no
    query, and no fragment. The error messages never repeat the value. The sinks and the SQM-LE
    reader of `seeingmon.hardware.sqm` share this check.
    """
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("use an http:// or https:// address, such as https://influx.example.org")
    if parts.username or parts.password:
        raise ValueError("do not put credentials in the endpoint; use token or password")
    if parts.query or parts.fragment:
        raise ValueError("leave the query and the fragment out of the endpoint")
    return value.rstrip("/")


def check_influx_connection(
    *,
    version: int,
    org: str | None,
    bucket: str | None,
    database: str | None,
    token: SecretStr | None,
    token_env: str | None,
    username: str | None,
    password: SecretStr | None,
    password_env: str | None,
) -> None:
    """Check the keys that an InfluxDB connection needs for its version, and its secrets.

    Version 1 needs a `database` and takes `username` and a password. Version 2 needs an `org`, a
    `bucket`, and takes a token. A secret has one source: a direct value or an environment
    variable. Raises `ValueError`. The messages never repeat a value. The sinks and the SQM-LE
    reader of `seeingmon.hardware.sqm` share this check.
    """
    if version == 1:
        if not database:
            raise ValueError("version 1 needs a database")
        if token is not None or token_env is not None:
            raise ValueError("version 1 has no token; use username and password")
    else:
        if not (org and bucket):
            raise ValueError("version 2 needs an org and a bucket")
        if username is not None or password is not None or password_env:
            raise ValueError("version 2 authenticates with a token, not a password")
    if token is not None and token_env is not None:
        raise ValueError("set token or token_env, not both")
    if password is not None and password_env is not None:
        raise ValueError("set password or password_env, not both")
    if (password is not None or password_env) and not username:
        raise ValueError("a password needs a username")


class _SinkBase(SectionModel):
    """The keys that every sink has. A number that an environment variable gives becomes text."""

    model_config = ConfigDict(frozen=True, extra="forbid", coerce_numbers_to_str=True)

    enabled: bool = Field(
        default=True, description="Set false to keep the table and skip the sink."
    )
    record_types: list[str] | None = Field(
        default=None,
        description="The record types to send. Leave it out to send every table record type.",
    )

    @field_validator("record_types")
    @classmethod
    def _check_record_types(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        from seeingmon.records.base import RECORD_TYPES

        for name in value:
            record = RECORD_TYPES.get(name)
            if record is None:
                raise ValueError(f"{name!r} is not a record type")
            if record.storage != "table":
                raise ValueError(f"{name!r} is stored in segment files, so no sink takes it")
        return value


class InfluxSinkConfig(_SinkBase):
    """An InfluxDB sink that writes line protocol over HTTP.

    Version 2 (`/api/v2/write`) needs `org`, `bucket`, and a `token`. InfluxDB 3 accepts the same
    call. Version 1 (`/write`) needs `database`, and it takes `username` and `password` when the
    server asks for them.
    """

    kind: Literal["influx"]
    endpoint: str = Field(
        description="The address of the server, such as https://influx.example.org."
    )
    version: Literal[1, 2] = Field(default=2, description="The write API: 1 or 2.")
    org: str | None = Field(default=None, description="The organization (version 2).")
    bucket: str | None = Field(default=None, description="The bucket (version 2).")
    database: str | None = Field(default=None, description="The database (version 1).")
    retention_policy: str | None = Field(
        default=None, description="The retention policy (version 1)."
    )
    token: SecretStr | None = Field(default=None, description="The API token (version 2).")
    token_env: str | None = Field(
        default=None, description="The name of the environment variable that holds the token."
    )
    username: str | None = Field(default=None, description="The user name (version 1).")
    password: SecretStr | None = Field(default=None, description="The password (version 1).")
    password_env: str | None = Field(
        default=None, description="The name of the environment variable that holds the password."
    )
    timeout_s: float = Field(
        default=10.0, gt=0, le=300, description="How long to wait for the server, in seconds."
    )
    verify_tls: bool = Field(
        default=True, description="Whether to check the certificate of the server."
    )
    max_batch_rows: int = Field(
        default=5000, ge=1, le=100_000, description="The most rows in one write request."
    )

    @field_validator("endpoint")
    @classmethod
    def _check_endpoint(cls, value: str) -> str:
        return check_endpoint(value)

    @model_validator(mode="after")
    def _check_version(self) -> Self:
        check_influx_connection(
            version=self.version,
            org=self.org,
            bucket=self.bucket,
            database=self.database,
            token=self.token,
            token_env=self.token_env,
            username=self.username,
            password=self.password,
            password_env=self.password_env,
        )
        return self


class TimescaleSinkConfig(_SinkBase):
    """A PostgreSQL or TimescaleDB sink. It needs the `timescale` extra (`psycopg`)."""

    kind: Literal["timescale"]
    host: str = Field(description="The host name of the server.")
    port: int = Field(default=5432, ge=1, le=65535, description="The port of the server.")
    database: str = Field(description="The name of the database.")
    user: str = Field(description="The user name.")
    password: SecretStr | None = Field(default=None, description="The password.")
    password_env: str | None = Field(
        default=None, description="The name of the environment variable that holds the password."
    )
    sslmode: SSL_MODES = Field(default="prefer", description="The PostgreSQL `sslmode`.")
    connect_timeout_s: float = Field(
        default=10.0, gt=0, le=300, description="How long to wait for a connection, in seconds."
    )
    hypertables: bool | None = Field(
        default=None,
        description=(
            "Whether to turn each table into a hypertable. Leave it out to detect the "
            "TimescaleDB extension and use it when it is there."
        ),
    )
    chunk_days: int = Field(
        default=7, ge=1, le=3650, description="The chunk interval of a hypertable, in days."
    )
    max_batch_rows: int = Field(
        default=1000, ge=1, le=100_000, description="The most rows in one transaction."
    )

    @model_validator(mode="after")
    def _check_password(self) -> Self:
        if self.password is not None and self.password_env is not None:
            raise ValueError("set password or password_env, not both")
        return self


SinkConfig = Annotated[InfluxSinkConfig | TimescaleSinkConfig, Field(discriminator="kind")]


class SinksSection(RootModel[dict[str, SinkConfig]]):
    """The `[sinks]` section: a map from the name of each sink to its configuration.

    Read it with `config.section("sinks", SinksSection)`. An empty or missing section means that
    the station keeps its results in the local store only.
    """

    @field_validator("root")
    @classmethod
    def _check_names(cls, value: dict[str, SinkConfig]) -> dict[str, SinkConfig]:
        for name in value:
            if not SINK_NAME.fullmatch(name):
                raise ValueError(
                    "a sink name starts with a lowercase letter and holds up to 32 lowercase "
                    "letters, digits, '_', and '-'"
                )
        return value


def resolve_credential(
    direct: SecretStr | None, variable: str | None, env: Mapping[str, str], what: str
) -> str | None:
    """Return a secret from its direct value or from the environment variable that holds it.

    `what` names the secret in an error message, such as `the token of sink influx`. Raises
    `ConfigError` when `variable` names an unset or empty variable. It never puts a secret in a
    message.
    """
    if variable:
        value = env.get(variable, "")
        if not value:
            raise ConfigError(
                f"{what} comes from the environment variable {variable}, which is not set"
            )
        return value
    if direct is not None:
        return direct.get_secret_value()
    return None
