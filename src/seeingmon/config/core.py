"""Load the layered configuration and read it by section.

Later layers override earlier ones:

1. The defaults: `config/default.toml`, then every `config/default.d/*.toml` in sorted file-name
   order. Each lane keeps its own defaults in its own file in `default.d/`.
2. The local file, `local/config.toml`. It is untracked and optional.
3. Environment variables, `SEEINGMON_<SECTION>__<KEY>=value` (see `seeingmon.config.layers`).

`load_config` merges the layers into a `Config`. A lane reads its part with
`config.section("name", Model)`, which validates the section with the lane's pydantic model.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from seeingmon import paths
from seeingmon.config import layers
from seeingmon.config.errors import ConfigError

if TYPE_CHECKING:
    from seeingmon.profile.models import Profile

DEFAULTS_FILE = "default.toml"
DEFAULTS_DIR = "default.d"
STATION_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"

ModelT = TypeVar("ModelT", bound=BaseModel)


class SectionModel(BaseModel):
    """A base class for a lane's configuration section.

    The model is frozen, and an unknown key is an error, so a misspelled key in
    `local/config.toml` fails loudly instead of being ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Config:
    """The merged configuration. Build it with `load_config`."""

    def __init__(self, data: Mapping[str, Any], *, profiles_dir: Path | None = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self._profiles_dir = profiles_dir
        self._profile: Profile | None = None
        self._check_top_level()

    def __repr__(self) -> str:
        return f"Config(sections={sorted(k for k, v in self._data.items() if isinstance(v, dict))})"

    def _check_top_level(self) -> None:
        station_id = self._data.get("station_id")
        if station_id is not None and not (
            isinstance(station_id, str) and re.fullmatch(STATION_ID_PATTERN, station_id)
        ):
            raise ConfigError(
                "station_id must be 1 to 64 letters, digits, '.', '_', or '-', starting with a "
                "letter or digit"
            )
        profile = self._data.get("profile")
        if profile is not None and not (isinstance(profile, str) and profile):
            raise ConfigError("profile must be a profile name or a path to a profile file")

    @property
    def station_id(self) -> str:
        """The short name that tags every record from this station."""
        station_id = self._data.get("station_id")
        if station_id is None:
            raise ConfigError(
                "station_id is not set: set it in local/config.toml or in SEEINGMON_STATION_ID"
            )
        return str(station_id)

    @property
    def profile_name(self) -> str:
        """The name of the configured profile (a file stem in `profiles/`, or a path)."""
        profile = self._data.get("profile")
        if profile is None:
            raise ConfigError(
                "profile is not set: set it in config/default.toml, in local/config.toml, "
                "or in SEEINGMON_PROFILE"
            )
        return str(profile)

    @property
    def profile(self) -> Profile:
        """The configured profile, loaded on first use.

        Raises `ProfileError` when the profile is missing or invalid.
        """
        if self._profile is None:
            from seeingmon.profile import load_profile

            self._profile = load_profile(self.profile_name, profiles_dir=self._profiles_dir)
        return self._profile

    def section(self, name: str, model: type[ModelT]) -> ModelT:
        """Validate one section with a lane's pydantic model, with its defaults applied.

        `name` is the table name, such as `"scheduler"`. A dotted name reads a nested table,
        such as `"sinks.influx"`. A missing section validates as an empty table, so the model's
        defaults apply, and a required field without a default is an error. Raises
        `ConfigError` with the problems listed, and never with a configured value.
        """
        table = self._table(name)
        try:
            return model.model_validate(copy.deepcopy(table))
        except ValidationError as error:
            raise ConfigError(_format_section_error(name, error)) from None

    def _table(self, name: str) -> dict[str, Any]:
        node: Any = self._data
        for part in name.split("."):
            if not isinstance(node, dict) or part not in node:
                return {}
            node = node[part]
        if not isinstance(node, dict):
            raise ConfigError(
                f"configuration section [{name}] must be a table, not a {type(node).__name__}"
            )
        return node

    def effective(self, redact: bool = True, *, omit_site: bool = False) -> dict[str, Any]:
        """The merged configuration as plain JSON types, for the `run` record.

        With `redact` set, the value of every key whose name contains token, password, secret,
        credential, or key, or that names a host, URL, address, command, path, folder, socket,
        pipe, or file (in any case, at any depth) becomes `"<redacted>"`. With `omit_site` set,
        the `site` section is left out, for output that must not carry site coordinates.
        TOML dates and times become ISO 8601 strings. The result is a copy.
        """
        result: dict[str, Any] = layers.jsonable(copy.deepcopy(self._data))
        if omit_site:
            result.pop("site", None)
        if redact:
            result = layers.redact(result)
        return result


def _format_section_error(name: str, error: ValidationError) -> str:
    lines = []
    for item in error.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"])
        message = item["msg"].removeprefix("Value error, ")
        if item["type"] == "missing":
            parts = [*name.split("."), *(str(part) for part in item["loc"])]
            if not any(part.isdigit() for part in parts):  # an array index has no variable
                variable = layers.ENV_PREFIX + layers.ENV_SEPARATOR.join(p.upper() for p in parts)
                message += f" (set it in local/config.toml or in {variable})"
        lines.append(f"  {location}: {message}" if location else f"  {message}")
    return f"invalid configuration section [{name}]:\n" + "\n".join(lines)


def load_config(
    *,
    config_dir: Path | str | None = None,
    local_file: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    profiles_dir: Path | str | None = None,
) -> Config:
    """Load the layers and merge them: defaults, then the local file, then the environment.

    By default the function reads the packaged `config/` directory (`default.toml` and
    `default.d/*.toml`), the file `local/config.toml` (see `seeingmon.paths.local_config_file`),
    and `os.environ`. Pass `config_dir`, `local_file`, `env`, or `profiles_dir` to read
    elsewhere. The local file is optional, and a missing one is ignored. To ignore both outer
    layers in a test, pass a `local_file` that does not exist and `env={}`.

    Raises `ConfigError` when `default.toml` is missing, a file is not valid TOML, or an
    environment variable is malformed.
    """
    directory = paths.config_dir() if config_dir is None else Path(config_dir)
    defaults = directory / DEFAULTS_FILE
    if not defaults.is_file():
        raise ConfigError(f"missing default configuration file {defaults}")
    stack = [layers.read_toml(defaults)]
    defaults_dir = directory / DEFAULTS_DIR
    if defaults_dir.is_dir():
        files = sorted(defaults_dir.glob("*.toml"), key=lambda path: path.name)
        stack += [
            layers.read_toml(path)
            for path in files
            if path.is_file() and not path.name.startswith(".")
        ]
    local = paths.local_config_file() if local_file is None else Path(local_file)
    if local.is_file():
        stack.append(layers.read_toml(local))
    stack.append(layers.env_overrides(os.environ if env is None else env))
    merged: dict[str, Any] = {}
    for layer in stack:
        merged = layers.deep_merge(merged, layer)
    return Config(merged, profiles_dir=None if profiles_dir is None else Path(profiles_dir))
