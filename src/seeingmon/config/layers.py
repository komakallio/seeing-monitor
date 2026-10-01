"""The building blocks of the configuration: files, environment variables, merging, redaction.

The functions are pure except `read_toml`, which reads one file. `load_config` in
`seeingmon.config.core` combines them in the order the architecture defines.

**Environment variables.** `SEEINGMON_<SECTION>__<KEY>=value` sets `[section] key = value`. A
double underscore nests one level deeper, so `SEEINGMON_SINKS__INFLUX__URL` sets
`[sinks.influx] url`. Keys are lowercase snake case, and the variable name is uppercase. A
single underscore stays part of the key. The value parses as a TOML scalar or array when it
parses (`60`, `0.5`, `true`, `[1, 2]`, `"quoted"`) and stays a string otherwise (`abc`). Use
quotes to force a string that looks like a number: `'"12345"'`. A value with a line break
always stays a string.
"""

from __future__ import annotations

import copy
import re
import tomllib
from collections.abc import Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from seeingmon.config.errors import ConfigError

ENV_PREFIX = "SEEINGMON_"
ENV_SEPARATOR = "__"
REDACTED = "<redacted>"
SECRET_WORDS = ("token", "password", "secret", "credential", "key")
# Deployment values (hosts, URLs, addresses, commands, and paths) say where a station runs,
# not how it computes, so they stay out of the effective configuration too. The match is on whole
# name parts, so `wind_direction_deg` is not a folder.
DEPLOYMENT_WORDS = frozenset(
    {
        "host",
        "hostname",
        "url",
        "uri",
        "endpoint",
        "address",
        "dsn",
        "command",
        "path",
        "paths",
        "dir",
        "dirs",
        "directory",
        "socket",
        "pipe",
        "file",
        "files",
    }
)

_ENV_SEGMENT = re.compile(r"[A-Z0-9]+(?:_[A-Z0-9]+)*")


def read_toml(path: Path) -> dict[str, Any]:
    """Read one TOML file. Raises `ConfigError` that names the file."""
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML ({error})") from error
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path}: cannot read the file ({error})") from error


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two configurations. The override wins.

    Tables merge key by key at every depth. Any other value, including an array, replaces
    the value in `base` as a whole. Neither argument changes.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_env_value(text: str) -> Any:
    """Parse an environment value as a TOML scalar or array, and fall back to the string."""
    if "\n" in text or "\r" in text:
        return text
    try:
        parsed = tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text
    if isinstance(parsed, dict):  # an inline table stays text, as a table is not a scalar
        return text
    return parsed


def env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """The configuration layer that the `SEEINGMON_*` variables in `env` define.

    Variables without the prefix are ignored. Raises `ConfigError` for a malformed name or for
    two variables that conflict, such as `SEEINGMON_SITE` and `SEEINGMON_SITE__ELEVATION_M`.
    """
    result: dict[str, Any] = {}
    owners: dict[tuple[str, ...], str] = {}  # the variable that set each key
    for name in sorted(env):
        if not name.startswith(ENV_PREFIX):
            continue
        segments = name[len(ENV_PREFIX) :].split(ENV_SEPARATOR)
        if not all(_ENV_SEGMENT.fullmatch(segment) for segment in segments):
            raise ConfigError(
                f"malformed environment variable {name}: write "
                f"{ENV_PREFIX}<SECTION>{ENV_SEPARATOR}<KEY> with uppercase letters, digits, "
                f"and single underscores, and {ENV_SEPARATOR!r} between levels"
            )
        path = tuple(segment.lower() for segment in segments)
        _set_path(result, path, parse_env_value(env[name]), name, owners)
    return result


def _set_path(
    tree: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
    variable: str,
    owners: dict[tuple[str, ...], str],
) -> None:
    node = tree
    for depth, key in enumerate(path[:-1], start=1):
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            other = owners[path[:depth]]
            raise ConfigError(f"environment variables {other} and {variable} conflict")
        node = child
    if isinstance(node.get(path[-1]), dict):
        other = next((v for k, v in owners.items() if k[: len(path)] == path), "another variable")
        raise ConfigError(f"environment variables {other} and {variable} conflict")
    node[path[-1]] = value
    owners[path] = variable


def is_secret_key(key: str) -> bool:
    """Whether a key name marks a secret: it has token, password, secret, credential, or key."""
    lowered = key.lower()
    return any(word in lowered for word in SECRET_WORDS)


def is_deployment_key(key: str) -> bool:
    """Whether a key name marks a deployment value: a part of the name is a host, URL, address,
    command, path, folder, socket, pipe, or file word, with parts split at every non-letter."""
    return any(part in DEPLOYMENT_WORDS for part in re.split(r"[^a-z0-9]+", key.lower()) if part)


def redact(value: Any) -> Any:
    """A copy of `value` with the value of every secret or deployment key replaced by `REDACTED`.

    A key is secret when its name contains token, password, secret, credential, or key, in any
    case. A key is a deployment key when a part of its name is a host, URL, address, command,
    path, folder, socket, pipe, or file word. The rule applies at any depth, and it replaces a
    table or an array under such a key as a whole.
    """
    if isinstance(value, Mapping):
        return {
            key: REDACTED
            if is_secret_key(str(key)) or is_deployment_key(str(key))
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def jsonable(value: Any) -> Any:
    """A copy of `value` in plain JSON types: TOML dates and times become ISO 8601 strings."""
    if isinstance(value, Mapping):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return value
