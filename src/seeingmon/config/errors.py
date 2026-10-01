"""Errors that the configuration package raises."""

from __future__ import annotations


class ConfigError(ValueError):
    """The configuration is missing, malformed, or invalid.

    The message names the file, the section, or the environment variable, and each wrong key.
    It never includes a configured value, so it is safe to log.
    """
