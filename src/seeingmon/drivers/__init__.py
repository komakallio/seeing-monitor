"""Camera drivers. The interface is in `seeingmon.drivers.base`."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from seeingmon.drivers.base import (
    CameraCaps,
    CameraConfigError,
    CameraDisconnectedError,
    CameraDriver,
    CameraError,
    CameraInfo,
    CameraStateError,
    CameraTimeoutError,
    RecoveryLevel,
)

if TYPE_CHECKING:
    from seeingmon.clock import Clock


def create_driver(name: str, *, profile: Any, clock: Clock, options: Any = None) -> CameraDriver:
    """Build the driver called `name`.

    The function imports `seeingmon.drivers.<name>` and calls its `create(profile=..., clock=...,
    options=...)`, so adding a driver changes no shared file. `name` is a plain module name such
    as `sim`, `asi`, or `replay`.
    """
    if not name.isidentifier() or name.startswith("_") or name == "base":
        raise ValueError(f"not a driver name: {name!r}")
    try:
        module = importlib.import_module(f"seeingmon.drivers.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"seeingmon.drivers.{name}":
            raise ValueError(f"unknown driver {name!r}") from exc
        raise
    driver: CameraDriver = module.create(profile=profile, clock=clock, options=options)
    return driver


__all__ = [
    "CameraCaps",
    "CameraConfigError",
    "CameraDisconnectedError",
    "CameraDriver",
    "CameraError",
    "CameraInfo",
    "CameraStateError",
    "CameraTimeoutError",
    "RecoveryLevel",
    "create_driver",
]
