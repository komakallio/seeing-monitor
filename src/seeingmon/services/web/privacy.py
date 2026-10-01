"""Keep private values out of the responses of the API.

The API serves data to the LAN, and the repository is public, so a response must never carry a
secret, the site coordinates, a private path, a host name, or an address. The configuration already
redacts the keys that name such values (`Config.effective`). This module is the second layer, for
the text that the first layer cannot see.

- `scrub_text` replaces a URL, a file path, and an IPv4 address inside free text, such as the text
  of an event or of an exception, with a marker.
- `scrub_json` applies `scrub_text` to every string of a JSON value.
- `public_config` reduces the effective configuration to the sections that describe how the station
  computes. It drops the sections that describe where and how one installation runs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

HIDDEN = "<hidden>"
MAX_DEPTH = 12

# Sections of the effective configuration that the API serves. Every other section describes the
# installation (paths, sinks, power, pins, addresses), so the API leaves it out.
PUBLIC_SECTIONS = (
    "profile",
    "station_id",
    "scheduler",
    "fastpath",
    "survey",
    "store",
    "services",
    "web",
)

# Keys that name a deployment value, beyond what `Config.effective` redacts.
_PRIVATE_KEY_PARTS = frozenset(
    {
        "serial",
        "user",
        "username",
        "login",
        "database",
        "org",
        "bucket",
        "driver_options",
        "headers",
        "latitude",
        "longitude",
        "elevation",
        "site",
    }
)

_URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+")
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>|]*")
_UNC_PATH = re.compile(r"\\\\[^\s\"'<>|]+")
_POSIX_PATH = re.compile(r"(?<![\w.:/~-])/(?:[\w.@+=,-]+/)+[\w.@+=,-]*")
_HOME_PATH = re.compile(r"(?<![\w.:/-])~/[\w.@+=,/-]*")
_IPV4 = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?!\w|\.\d)")


def scrub_text(text: str) -> str:
    """Replace URLs, file paths, and IPv4 addresses in `text` with `<hidden>`."""
    for pattern in (_URL, _WINDOWS_PATH, _UNC_PATH, _POSIX_PATH, _HOME_PATH, _IPV4):
        text = pattern.sub(HIDDEN, text)
    return text


def scrub_json(value: Any, *, _depth: int = 0) -> Any:
    """A copy of a JSON value with `scrub_text` applied to every string, keys excluded."""
    if _depth > MAX_DEPTH:
        return HIDDEN
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, Mapping):
        return {key: scrub_json(item, _depth=_depth + 1) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [scrub_json(item, _depth=_depth + 1) for item in value]
    return value


def _private_key(key: str) -> bool:
    lowered = key.lower()
    parts = {part for part in re.split(r"[^a-z0-9]+", lowered) if part}
    return lowered in _PRIVATE_KEY_PARTS or bool(parts & _PRIVATE_KEY_PARTS)


def _public_value(value: Any, depth: int) -> Any:
    if depth > MAX_DEPTH:
        return HIDDEN
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item, depth + 1)
            for key, item in value.items()
            if not _private_key(str(key))
        }
    if isinstance(value, list | tuple):
        return [_public_value(item, depth + 1) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value


def public_config(effective: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the effective configuration that the API may serve.

    `effective` is the result of `Config.effective(redact=True, omit_site=True)`. The function keeps
    the sections in `PUBLIC_SECTIONS`, drops the keys that name a deployment value (serials, users,
    databases, `driver_options`), and replaces text that looks like a path, a URL, or an address.
    """
    return {
        name: _public_value(effective[name], 0) for name in PUBLIC_SECTIONS if name in effective
    }
