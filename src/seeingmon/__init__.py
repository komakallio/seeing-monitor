"""Seeing and sky-quality monitor for a fixed camera that points at Polaris."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("seeingmon")
except PackageNotFoundError:  # a source tree without an installed distribution
    __version__ = "0+unknown"

__all__ = ["__version__"]
