"""Build the sinks of a station from its configuration.

`build_sinks(config)` reads the `[sinks]` section (`seeingmon.sinks.config` describes it), skips
the sinks with `enabled = false`, resolves each secret, and returns the sinks in the order of the
file. Pass them to a `Forwarder`. The function never prints or logs a secret, and an error names a
sink and a key but never a value.

```python
from seeingmon.config import load_config
from seeingmon.sinks.factory import build_sinks

sinks = build_sinks(load_config())
```

A sink that needs a driver checks it here, when the service starts, so a missing `psycopg` fails
at once with a message that names the `timescale` extra.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from types import ModuleType

from seeingmon.config import Config
from seeingmon.sinks.base import Sink
from seeingmon.sinks.config import (
    InfluxSinkConfig,
    SinksSection,
    TimescaleSinkConfig,
    resolve_credential,
)
from seeingmon.sinks.influx import InfluxSink
from seeingmon.sinks.timescale import TimescaleSink, make_psycopg_connect


def build_sinks(
    config: Config,
    *,
    env: Mapping[str, str] | None = None,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> list[Sink]:
    """Build the enabled sinks of the `[sinks]` section.

    `env` is where `token_env` and `password_env` look, and it defaults to `os.environ`. A test
    passes a dict. `import_module` imports `psycopg` for a `timescale` sink, and a test replaces
    it. Raises `ConfigError` for an invalid section, for a variable that is not set, and for a
    missing driver.
    """
    environment = os.environ if env is None else env
    sinks: list[Sink] = []
    for name, settings in config.section("sinks", SinksSection).root.items():
        if not settings.enabled:
            continue
        if isinstance(settings, InfluxSinkConfig):
            sinks.append(_influx(name, settings, environment))
        elif isinstance(settings, TimescaleSinkConfig):
            sinks.append(_timescale(name, settings, environment, import_module))
    return sinks


def _influx(name: str, settings: InfluxSinkConfig, env: Mapping[str, str]) -> InfluxSink:
    token = resolve_credential(settings.token, settings.token_env, env, f"the token of sink {name}")
    password = resolve_credential(
        settings.password, settings.password_env, env, f"the password of sink {name}"
    )
    return InfluxSink(name, settings, token=token, password=password)


def _timescale(
    name: str,
    settings: TimescaleSinkConfig,
    env: Mapping[str, str],
    import_module: Callable[[str], ModuleType],
) -> TimescaleSink:
    password = resolve_credential(
        settings.password, settings.password_env, env, f"the password of sink {name}"
    )
    connect = make_psycopg_connect(settings, password, import_module=import_module)
    return TimescaleSink.from_config(name, settings, connect)
