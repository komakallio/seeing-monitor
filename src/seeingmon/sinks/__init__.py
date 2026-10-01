"""Result sinks. The interface is in `seeingmon.sinks.base`.

Import the classes from here, for example `from seeingmon.sinks import Forwarder`. The package
loads each name on first use, so importing `seeingmon.sinks` stays fast.

- `base`: the `Sink` protocol, `SinkError`, and `StoredRow` (the contract).
- `forwarder`: `Forwarder`, which sends the rows of the store to every sink with cursors, retries,
  and backoff.
- `config`: the `[sinks.<name>]` configuration models.
- `influx`: `InfluxSink`, an InfluxDB line-protocol sink for versions 1 and 2.
- `timescale`: `TimescaleSink`, a PostgreSQL or TimescaleDB sink over a DB-API connection.
- `factory`: `build_sinks`, which builds the sinks of the `[sinks]` configuration.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from seeingmon.sinks.base import Sink, SinkError, StoredRow

if TYPE_CHECKING:
    from seeingmon.sinks.forwarder import Forwarder, ForwardReport, SinkPass, SinkStatus

# The module that defines each lazily loaded name.
_EXPORTS: dict[str, str] = {
    "InfluxSinkConfig": "config",
    "SinksSection": "config",
    "TimescaleSinkConfig": "config",
    "resolve_credential": "config",
    "ForwardReport": "forwarder",
    "Forwarder": "forwarder",
    "SinkPass": "forwarder",
    "SinkStatus": "forwarder",
    "InfluxSink": "influx",
    "TimescaleSink": "timescale",
    "make_psycopg_connect": "timescale",
    "build_sinks": "factory",
}

__all__ = [
    "ForwardReport",
    "Forwarder",
    "InfluxSink",
    "InfluxSinkConfig",
    "Sink",
    "SinkError",
    "SinkPass",
    "SinkStatus",
    "SinksSection",
    "StoredRow",
    "TimescaleSink",
    "TimescaleSinkConfig",
    "build_sinks",
    "make_psycopg_connect",
    "resolve_credential",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
