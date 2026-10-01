"""Layered configuration: defaults, a local file, and environment variables.

    from seeingmon.config import SectionModel, load_config

    class SchedulerConfig(SectionModel):
        window_s: float = 120.0

    config = load_config()
    scheduler = config.section("scheduler", SchedulerConfig)
    profile = config.profile

See `seeingmon.config.core` for the layers and `seeingmon.config.layers` for the environment
variable scheme.
"""

from __future__ import annotations

from seeingmon.config.core import Config, SectionModel, load_config
from seeingmon.config.errors import ConfigError
from seeingmon.config.layers import REDACTED

__all__ = ["REDACTED", "Config", "ConfigError", "SectionModel", "load_config"]
